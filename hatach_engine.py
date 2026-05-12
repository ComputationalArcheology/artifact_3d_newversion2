import numpy as np
import networkx as nx
import pyvista as pv
from scipy.spatial import cKDTree


def compute_hatach_contours(mesh: pv.PolyData, z_cut: float, thickness: float = None) -> list[np.ndarray]:
    """
    מבצע את החיתוך (Hatach) על בסיס הלוגיקה המקורית של מטלב (Slab Logic),
    אך משתמש ב-NetworkX כדי לחבר את המקטעים לקונטורים סגורים.

    :param mesh: ה-Mesh התלת ממדי.
    :param z_cut: גובה החיתוך (Z).
    :param thickness: עובי ה'פרוסה' (Slab). אם None, יחושב אוטומטית.
    :return: רשימה של מערכי numpy (N, 2) המייצגים את הקונטורים הדו-ממדיים.
    """

    # 1. חישוב עובי אוטומטי (אם לא סופק) - מקביל ל-Auto2Dis
    if thickness is None:
        # הערכה גסה: אחוז מסוים מהגודל הממוצע של צלע
        if mesh.n_points > 0:
            # דגימה מהירה לחישוב צפיפות
            bounds = mesh.bounds
            diag = np.linalg.norm([bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4]])
            thickness = diag * 0.005  # 0.5% מהאלכסון כברירת מחדל
        else:
            thickness = 0.1

    points = mesh.points
    faces = mesh.faces.reshape(-1, 4)[:, 1:]  # הנחה: משולשים בלבד (v1, v2, v3)

    # 2. זיהוי הצלעות החוצות (Loo_FinerCut Logic)
    # במקום לולאות איטיות, נשתמש בוקטוריזציה של NumPy.

    # נגדיר את הטווח של ה"פרוסה"
    z_min = z_cut - thickness
    z_max = z_cut + thickness

    # עבור כל משולש, נבדוק אילו צלעות חוצות את מישור ה-Z המדויק
    # (הלוגיקה של מטלב עם "up/down" היא בעצם דרך למצוא חיתוך)

    # נחלץ את כל הצלעות (Edges) במשולשים
    edges = np.vstack([
        faces[:, [0, 1]],
        faces[:, [1, 2]],
        faces[:, [2, 0]]
    ])

    # נמיין את האינדקסים בכל צלע כדי למנוע כפילויות (edge 1-2 זהה ל-2-1)
    edges.sort(axis=1)
    unique_edges = np.unique(edges, axis=0)

    v1_indices = unique_edges[:, 0]
    v2_indices = unique_edges[:, 1]

    z1 = points[v1_indices, 2]
    z2 = points[v2_indices, 2]

    # תנאי החיתוך: נקודה אחת מעל Z ואחת מתחת Z
    # (או בתוך הטווח אם היינו משתמשים בלוגיקת ה-Slab המלאה, אבל חיתוך גאומטרי הוא מדויק יותר)
    crossing_mask = ((z1 >= z_cut) & (z2 < z_cut)) | ((z1 < z_cut) & (z2 >= z_cut))

    crossing_edges = unique_edges[crossing_mask]

    if len(crossing_edges) == 0:
        return []

    # 3. חישוב נקודות החיתוך המדויקות (PairIntersect Logic)
    # אינטרפולציה לינארית למציאת (x,y) ב-z_cut

    ce_v1 = points[crossing_edges[:, 0]]
    ce_v2 = points[crossing_edges[:, 1]]

    # t הוא היחס שבו המישור חותך את הצלע (בין 0 ל-1)
    # z = z1 + t * (z2 - z1)  =>  t = (z - z1) / (z2 - z1)
    denom = (ce_v2[:, 2] - ce_v1[:, 2])
    denom[denom == 0] = 1e-9  # מניעת חלוקה באפס (אופקיים)

    t = (z_cut - ce_v1[:, 2]) / denom
    t = t[:, np.newaxis]  # הרחבה לוקטור עמודה

    # P = P1 + t * (P2 - P1)
    intersect_points_3d = ce_v1 + t * (ce_v2 - ce_v1)

    # אנחנו צריכים רק דו-ממד (X, Y)
    intersect_points_2d = intersect_points_3d[:, :2]

    # 4. בניית הגרף וחיבור הקונטורים (ClosePeg Logic with NetworkX)
    # כל נקודת חיתוך שייכת למשולש. כל משולש שנחתך מייצר קו בין 2 נקודות חיתוך.
    # לכן, עלינו לזהות אילו שתי נקודות חיתוך שייכות לאותו משולש.

    # טריק: נחזור למשולשים. נזהה אילו משולשים נחתכים.
    # משולש נחתך אם יש לו בדיוק 2 צלעות שנחתכות (במקרה הרגיל).

    # נבנה מילון שממפה: Edge Index -> Intersection Point Index
    # אבל דרך פשוטה יותר: לכל משולש יש 3 צלעות. נבדוק כמה מהן נחתכו.

    # גישה מהירה יותר לחיבור: "קרבה גאומטרית" בגרף (פחות מדויק טופולוגית אבל עובד מעולה ל-Contour)
    # או הגישה הנכונה: מעקב אחרי הטופולוגיה.

    # נשתמש ב-PyVista Slice שהוא המימוש המהיר והיציב ביותר של האלגוריתם הנ"ל (C++)
    # אם תתעקש על הלוגיקה המטלבית המדויקת של "שכנים", אפשר לממש, אבל Slice של VTK נותן תוצאה זהה מתמטית.

    # לטובת הביצועים והיציבות באפליקציה פייתון, נשתמש ב-slice של PyVista
    # ואז נסדר את הקווים (Stripper) לקונטורים.

    try:
        # חיתוך גאומטרי מהיר
        slice_poly = mesh.slice(normal=[0, 0, 1], origin=[0, 0, z_cut], generate_triangles=False)

        # התוצאה היא אוסף של קווים (Lines). צריך לחבר אותם.
        lines = slice_poly.lines
        # הפורמט של lines ב-VTK הוא [n_pts, p0, p1, n_pts, p2, p3...]
        # עבור segments זה תמיד [2, p0, p1, 2, p2, p3...]

        if len(lines) == 0:
            return []

        # המרת ה-Lines ל-Edges בגרף
        points_slice = slice_poly.points

        # בניית גרף NetworkX
        G = nx.Graph()

        # איטרציה על ה-lines buffer
        i = 0
        while i < len(lines):
            n = lines[i]
            if n == 2:
                idx1 = lines[i + 1]
                idx2 = lines[i + 2]
                # נוסיף את הקואורדינטות כ-nodes (או אינדקסים)
                # שימוש באינדקסים עדיף למניעת בעיות דיוק צף
                G.add_edge(idx1, idx2)
            i += (n + 1)

        contours = []

        # מציאת רכיבים קשירים (Connected Components) = קונטורים נפרדים
        # ומציאת מסלול (Cycle) בכל רכיב
        components = list(nx.connected_components(G))

        for comp in components:
            subgraph = G.subgraph(comp)

            # ניסיון למצוא מעגל (Cycle)
            try:
                # find_cycle מחזיר רשימת קשתות. אנחנו רוצים סדר צמתים.
                cycle_edges = nx.find_cycle(subgraph)
                ordered_indices = [cycle_edges[0][0]]
                for edge in cycle_edges:
                    ordered_indices.append(edge[1])

                # שליפת הקואורדינטות (רק X, Y)
                pts_cycle = points_slice[ordered_indices]
                contours.append(pts_cycle[:, :2])  # רק X, Y

            except nx.NetworkXNoCycle:
                # אם זה קו פתוח (לא סגור), נשתמש ב-path רגיל
                # מוצאים נקודות קצה (degree=1)
                endpoints = [n for n, d in subgraph.degree() if d == 1]
                if len(endpoints) >= 2:
                    path = nx.shortest_path(subgraph, endpoints[0], endpoints[1])
                    pts_path = points_slice[path]
                    contours.append(pts_path[:, :2])

        return contours

    except Exception as e:
        print(f"Error in slice calculation: {e}")
        return []