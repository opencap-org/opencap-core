import argparse
import logging
import time
import traceback

import requests

from utils import getTrialJson, makeRequestWithRetry
from utilsAPI import getAPIURL
from utilsAuth import getToken
from utilsServer import processTrial


logging.basicConfig(format="[%(asctime)s] [%(levelname)s] %(message)s",
                    level=logging.INFO,
                    datefmt="%Y-%m-%d %H:%M:%S")

API_URL = getAPIURL()
API_TOKEN = getToken()


def get_trial_type(trial):
    if trial["name"] == "calibration":
        return "calibration"
    if trial["name"] == "neutral":
        return "static"
    return "dynamic"


def get_next_trial(worker_type="all"):
    queue_path = "trials/dequeue/?workerType={}".format(worker_type)
    response = requests.get(
        "{}{}".format(API_URL, queue_path),
        headers={"Authorization": "Token {}".format(API_TOKEN)})

    if response.status_code == 404:
        return None

    response.raise_for_status()
    return response.json()


def patch_trial_status(trial_id, status):
    trial_url = "{}{}{}/".format(API_URL, "trials/", trial_id)
    makeRequestWithRetry(
        "PATCH",
        trial_url,
        data={"status": status},
        headers={"Authorization": "Token {}".format(API_TOKEN)})


def process_local_trial(trial, skip_status_patch=False):
    trial_type = get_trial_type(trial)
    logging.info("Processing trial %s from session %s as %s.",
                 trial["id"], trial["session"], trial_type)

    try:
        processTrial(trial["session"], trial["id"], trial_type=trial_type,
                     isDocker=False, deleteLocalFolder=False)
    except Exception:
        traceback.print_exc()
        if not skip_status_patch:
            patch_trial_status(trial["id"], "error")
        raise

    if not skip_status_patch:
        patch_trial_status(trial["id"], "done")

    logging.info("Finished trial %s.", trial["id"])


def main():
    parser = argparse.ArgumentParser(
        description="Local polling trial processor for sandbox testing.")
    parser.add_argument("--trial-id", help="Process a specific trial once instead of polling.")
    parser.add_argument("--worker-type", default="all",
                        help="Worker type to use when dequeuing. Defaults to all.")
    parser.add_argument("--sleep-seconds", type=float, default=2,
                        help="Seconds to sleep when no queued trial is found.")
    parser.add_argument("--skip-status-patch", action="store_true",
                        help="Do not PATCH the trial status to done/error.")
    args = parser.parse_args()

    if args.trial_id:
        process_local_trial(getTrialJson(args.trial_id), args.skip_status_patch)
        return

    while True:
        try:
            trial = get_next_trial(args.worker_type)
        except Exception:
            traceback.print_exc()
            time.sleep(args.sleep_seconds)
            continue

        if trial is None:
            logging.info("No queued trial found for workerType=%s.", args.worker_type)
            time.sleep(args.sleep_seconds)
            continue

        try:
            process_local_trial(trial, args.skip_status_patch)
        except Exception:
            logging.info("Trial %s failed. Continuing local polling loop.", trial["id"])


if __name__ == "__main__":
    main()
