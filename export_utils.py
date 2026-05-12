# src/artifact_app/processing/export_utils.py
import pandas as pd
import numpy as np
from datetime import datetime
from .process_object_measurements import ProcessObjectMeasurements


def export_measurements_to_excel(
        measurements: ProcessObjectMeasurements,
        object_name: str,
        filepath: str,
        actual_com: np.ndarray = None
):
    """
    משחזר את הלוגיקה מ-'write_for_GUI.m' של MATLAB לשמירת הנתונים לאקסל.
    """

    DD0 = measurements.profile_xz.matlab_data
    DD1 = measurements.profile_yz.matlab_data
    bbox = measurements.bind3box

    if DD0 is None or DD1 is None:
        return False

    data = []

    data.append(("Name", object_name))
    data.append(("Date", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    data.append(("Volume", measurements.mesh_volume))

    data.append(("Length (cordh)", DD1.cordh))
    data.append(("Width (cordw)", DD1.cordw))
    data.append(("Thickness (cordw_xz)", DD0.cordw))

    data.append(("Virtual binding box length", DD1.calh))
    data.append(("Virtual binding box width", DD1.calw))
    data.append(("Virtual binding box thickness", DD0.calw))
    data.append(("Virtual binding box volume", bbox.volume))

    data.append(("Length of higher touch point (Width view)", DD1.hcah))
    data.append(("Length of higher touch point (Thickness view)", DD0.hcah))

    data.append(("Width at half length", DD1.cordwhh))
    data.append(("Thickness at half length", DD0.cordwhh))
    data.append(("Width at 1/5 length", DD1.cord20p))
    data.append(("Thickness at 1/5 length", DD0.cord20p))
    data.append(("Width at 4/5 length", DD1.cord80p))
    data.append(("Thickness at 4/5 length", DD0.cord80p))

    c_box = bbox.center_world if bbox.center_world is not None else np.array([0.0, 0.0, 0.0])
    x_cen, y_cen, z_cen = c_box[0], c_box[1], c_box[2]

    data.append(("Center BB X", x_cen))
    data.append(("Center BB Y", y_cen))
    data.append(("Center BB Z", z_cen))

    norm_center = np.sqrt(x_cen ** 2 + y_cen ** 2 + z_cen ** 2)
    data.append(("Norm of center point", norm_center))

    # חישוב ושמירת מרכז המסה והמרחק שלו ממרכז ה-Bounding Box (אם סופק)
    if actual_com is not None and bbox.center_world is not None:
        dist_to_com = np.linalg.norm(bbox.center_world - actual_com)
        data.append(("Center of Mass X", actual_com[0]))
        data.append(("Center of Mass Y", actual_com[1]))
        data.append(("Center of Mass Z", actual_com[2]))
        data.append(("Distance BBox Center to CoM", dist_to_com))

    if bbox.dims and bbox.dims['dx'] > 0:
        pass

    df = pd.DataFrame(data, columns=["Parameter", "Value"])

    try:
        df.to_excel(filepath, index=False)
        return True
    except Exception as e:
        print(f"Error saving Excel: {e}")
        return False


def export_batch_to_excel(batch_data: list[dict], filepath: str):
    """
    מייצא רשימה של מדידות לקובץ אקסל אחד מרוכז.
    """
    try:
        df = pd.DataFrame(batch_data)
        df.to_excel(filepath, index=False)
        return True
    except Exception as e:
        print(f"Error saving batch Excel: {e}")
        return False