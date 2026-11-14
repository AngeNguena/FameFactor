
import pandas as pd

   # Convert a column to datetime format.
   

def convert_dates(df, date_column):
   

    df = df.copy()

    df[date_column] = pd.to_datetime(
        df[date_column],
        errors="coerce"
    )

    return df


def clean_videos(df):
    """
    Clean the video dataset and create data quality flags.
    """
    df = df.copy()

    # Convert publication dates
    df = convert_dates(df, "create_date")

    # Calculate the total number of hashtags
    df["calculated_hashtag_count"] = df["hashtags"].apply(len)

    
    df["unique_hashtag_count"] = df["hashtags"].apply(
        lambda hashtags: len({
            hashtag["hashtag_name"]
            for hashtag in hashtags
            if pd.notna(hashtag["hashtag_name"])
        })
)


    # Flag missing hashtag counts
    df["hashtag_count_was_missing"] = (
        df["hashtag_count"].isna()
    )

    # Flag inconsistent hashtag counts
    df["hashtag_count_mismatch"] = (
        df["hashtag_count"].notna()
        & (
            df["hashtag_count"]
            != df["unique_hashtag_count"]
        )
    )

    # Recover missing hashtag counts
    df["hashtag_count"] = df["hashtag_count"].fillna(
        df["unique_hashtag_count"]
    )

    return df



def clean_engagement(df):
    """
    Clean engagement data and flag possible metric resets.
    """
    # Convert observation dates
    df = convert_dates(df, "date")

    # Sort observations chronologically
    df = df.sort_values(["video_id", "date"])

    engagement_columns = [
        "play_count",
        "like_count",
        "comment_count",
        "share_count",
        "collect_count",
        "download_count",
        "whatsapp_share_count"
    ]

    # Initialize the number of sudden drops
    drop_count = pd.Series(0, index=df.index)

    # Count metrics dropping from positive values to zero
    for column in engagement_columns:

        previous = df.groupby("video_id")[column].shift(1)

        drops_to_zero = (
            (previous > 0)
            & (df[column] == 0)
        )

        drop_count += drops_to_zero.astype(int)

    # Identify views dropping to zero
    previous_views = df.groupby("video_id")["play_count"].shift(1)

    views_drop_to_zero = (
        (previous_views > 0)
        & (df["play_count"] == 0)
    )

    # Apply the final anomaly detection rules
    df["possible_engagement_reset"] = (
        (drop_count >= 3)
        | (views_drop_to_zero & (drop_count >= 2))
    )

    return df


def clean_creators(df):
    """
    Clean creator data and flag possible follower anomalies.
    """
    # Convert observation dates
    df = convert_dates(df, "date")

    # Flag missing follower and following counts
    df["follower_data_missing"] = (
        df["follower_count"].isna()
        | df["following_count"].isna()
    )

    # Flag missing video counts
    df["video_count_missing"] = (
        df["video_count"].isna()
    )

    # Remove enterprise_verified if entirely missing
    if df["enterprise_verified"].isna().all():
        df = df.drop(columns=["enterprise_verified"])

    # Sort observations chronologically
    df = df.sort_values(["author_id", "date"])

    # Get previous and next follower counts
    previous_followers = df.groupby(
        "author_id"
    )["follower_count"].shift(1)

    next_followers = df.groupby(
        "author_id"
    )["follower_count"].shift(-1)

    # Calculate percentage decrease
    follower_drop_pct = (
        (previous_followers - df["follower_count"])
        / previous_followers.replace(0, float("nan"))
    )

    # Calculate recovery difference
    recovery_difference = (
        (next_followers - previous_followers).abs()
        / previous_followers.replace(0, float("nan"))
    )

    # Get previous and next observation dates
    previous_date = df.groupby(
        "author_id"
    )["date"].shift(1)

    next_date = df.groupby(
        "author_id"
    )["date"].shift(-1)

    # Flag possible temporary follower anomalies
    df["possible_follower_anomaly"] = (
        (follower_drop_pct >= 0.20)
        & (recovery_difference <= 0.05)
        & ((df["date"] - previous_date).dt.days == 1)
        & ((next_date - df["date"]).dt.days == 1)
    )

    return df
