import os
import xml.etree.ElementTree as ET


def update_xml_paths(
    template_xml_path: str,
    results_directory: str,
    model_file: str,
    marker_file: str,
    output_motion_file: str,
    output_xml_path: str,
):
    """
    Load an IK setup template, update paths, and save to output_xml_path.
    The original template file is not modified unless output_xml_path is the same path.
    """
    tree = ET.parse(template_xml_path)
    root = tree.getroot()

    ik_tool = root.find(".//InverseKinematicsTool")
    if ik_tool is None:
        raise ValueError("InverseKinematicsTool not found in the XML file.")

    ik_tool.find("results_directory").text = results_directory
    ik_tool.find("model_file").text = model_file
    ik_tool.find("marker_file").text = marker_file
    ik_tool.find("output_motion_file").text = output_motion_file

    os.makedirs(os.path.dirname(output_xml_path) or ".", exist_ok=True)
    tree.write(output_xml_path, encoding="utf-8", xml_declaration=True)
    print(f"Read template: {template_xml_path}")
    print(f"Wrote updated XML to: {output_xml_path}")


def main():
    template_path = r"G:\Shared drives\Stanford Football\March_2\subject5\OpenSimData\OpenPose_default\3-cameras\Kinematics_NoSync\Setup_IK_ID5_S7_sprint_LSTM.xml"

    # Destination folder for the updated setup (and where IK results should point).
    output_dir = r"G:\Shared drives\Stanford Football\AnalysisCompare\LengthFilt"

    model_file = r"G:\Shared drives\Stanford Football\March_2\subject5\OpenSimData\OpenPose_default\3-cameras\Model\LaiUhlrich2022_scaled_no_patella.osim"
    marker_file = r"G:\Shared drives\Stanford Football\March_2\subject5\MarkerData\OpenPose_default\3-cameras\PreAugmentation\ID5_S7_sprintNoSync.trc"

    trial_stem = os.path.splitext(os.path.basename(marker_file))[0]
    output_motion_file = os.path.join(output_dir, trial_stem + ".mot")

    output_xml_path = os.path.join(output_dir, os.path.basename(template_path))

    update_xml_paths(
        template_path,
        output_dir,
        model_file,
        marker_file,
        output_motion_file,
        output_xml_path,
    )


if __name__ == "__main__":
    main()
