# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 13:37:39 2022

@author: suhlr
"""
import os
import boto3
import requests
import logging

from decouple import config
from datetime import datetime, timedelta

def getAPIURL():
    if 'API_URL' not in globals():
        global API_URL
        try: # look in environment file
            API_URL = config("API_URL")
        except: # default
            API_URL = "https://api.opencap.ai/"
    
    if API_URL[-1] != '/':
        API_URL= API_URL + '/'

    return API_URL

def getWorkerType():
    try: # look in environment file
        workerType = config("WORKER_TYPE")
    except: # default
        workerType = "all"
    
    return workerType

def getStatusEmails():
    import json
    emailInfo = {}
    try:
        emailInfo['fromEmail'] = config("STATUS_EMAIL_FROM")
        emailInfo['password'] = config("STATUS_EMAIL_FROM_PW")
        emailInfo['toEmails'] = json.loads(config("STATUS_EMAIL_TO"))
    except:
        emailInfo = None
    try:
        emailInfo['ip'] = config("STATUS_EMAIL_IP")   
    except:
        pass
    
    return emailInfo

def getErrorLogBool():
    return config('ERROR_LOG', default=False, cast=bool)

def getAppPullWaitTimeAndJitter():
    time = config('APP_PULL_WAIT_TIME', default=5.0, cast=float)
    jitter = config('APP_PULL_WAIT_TIME_JITTER', default=1.0, cast=float)

    return time, jitter

def getLogLevel():
    log_level_str = config('LOG_LEVEL', default='INFO')
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)

    return log_level

def getASInstance():
    try:
        # Check if the ECS_CONTAINER_METADATA_FILE environment variable exists
        ecs_metadata_file = os.getenv('ECS_CONTAINER_METADATA_FILE')
        if ecs_metadata_file:
            if os.path.isfile(ecs_metadata_file):
                return True
            else:
                return False
        else:
            return False
    except Exception as e:
        return False

def get_metric_average(namespace, metric_name, start_time, end_time, period):
    """
    Fetch the average value of a specific metric from AWS CloudWatch.

    Parameters:
    - namespace (str): The namespace for the metric data.
    - metric_name (str): The name of the metric.
    - start_time (datetime): Start time for the data retrieval.
    - end_time (datetime): End time for the data retrieval.
    - period (int): The granularity, in seconds, of the data points returned.
    """
    client = boto3.client('cloudwatch', region_name='us-west-2')
    response = client.get_metric_statistics(
        Namespace=namespace,
        MetricName=metric_name,
        StartTime=start_time,
        EndTime=end_time,
        Period=period,
        Statistics=['Average']  # Correctly specifying 'Average' here
    )
    return response

def get_number_of_pending_trials(period=60):
    # Time range setup for the last 1 minute
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(minutes=1)

    # Fetch the metric data
    namespace = 'Custom/opencap-dev'  # or 'Custom/opencap' for production
    metric_name = 'opencap_trials_pending'
    stats = get_metric_average(
        namespace, metric_name, start_time, end_time, period)

    if stats['Datapoints']:
        average = stats['Datapoints'][0]['Average']
    else:
        # Maybe raise an exception or do nothing to have control-loop retry this call later
        return None

    return average

def get_instance_id():
    """Retrieve the instance ID from EC2 metadata."""
    response = requests.get("http://169.254.169.254/latest/meta-data/instance-id")
    return response.text

def get_auto_scaling_group_name(instance_id):
    """Retrieve the Auto Scaling Group name using the instance ID."""
    client = boto3.client('autoscaling', region_name='us-west-2')
    response = client.describe_auto_scaling_instances(InstanceIds=[instance_id])
    asg_name = response['AutoScalingInstances'][0]['AutoScalingGroupName']
    return asg_name

def set_instance_protection(instance_id, asg_name, protect):
    """Set or remove instance protection."""
    client = boto3.client('autoscaling', region_name='us-west-2')
    client.set_instance_protection(
        InstanceIds=[instance_id],
        AutoScalingGroupName=asg_name,
        ProtectedFromScaleIn=protect
    )

def unprotect_current_instance():
    instance_id = get_instance_id()
    asg_name = get_auto_scaling_group_name(instance_id)
    set_instance_protection(instance_id, asg_name, protect=False)
