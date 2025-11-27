from typing import List

def cluster_by_time_gaps(
    timestamps: List[float],
    time_gap_max: float,
    min_cluster_size: int
) -> List[List[int]]:
    """
    Cluster timestamps by time gaps (1D DBSCAN-like).
    Returns bursts as lists of indices.
    """

    if not timestamps:
        return []

    bursts = []
    current = [0]

    for i in range(1, len(timestamps)):
        # מרחק בזמן בין תמונה לזו שלפניה
        gap = timestamps[i] - timestamps[i - 1]

        # אם שייך לאותו burst
        if gap <= time_gap_max:
            current.append(i)

        else:
            # אם הקלאסטר גדול מספיק — נשמור
            if len(current) >= min_cluster_size:
                bursts.append(current)
            # נתחיל burst חדש
            current = [i]

    # טיפול ב־burst האחרון
    if len(current) >= min_cluster_size:
        bursts.append(current)

    return bursts
