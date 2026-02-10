def commodity_query(start_date: str, end_date: str, commodity: str) -> str:
    return f"""
    SELECT
        PARSE_DATE('%Y%m%d', SUBSTR(CAST(DATE AS STRING), 1, 8)) AS date,
        COUNT(*) AS volume,
        AVG(CAST(V2Tone AS FLOAT64)) AS tone,
        STDDEV(CAST(V2Tone AS FLOAT64)) AS tone_std,
        COUNTIF(CAST(V2Tone AS FLOAT64) < -2.0) AS negative_count,
        COUNTIF(CAST(V2Tone AS FLOAT64) > 2.0) AS positive_count
    FROM
        `gdelt-bq.gdeltv2.gkg_partitioned`
    WHERE
        _PARTITIONTIME BETWEEN TIMESTAMP('{start_date}') AND TIMESTAMP('{end_date}')
        AND PARSE_DATE('%Y%m%d', SUBSTR(CAST(DATE AS STRING), 1, 8))
            BETWEEN '{start_date}' AND '{end_date}'
        AND (
            LOWER(V2Themes) LIKE '%{commodity}%'
            OR LOWER(DocumentIdentifier) LIKE '%{commodity}%'
        )
    GROUP BY date
    ORDER BY date
    """  # noqa: S608


def agriculture_query(start_date: str, end_date: str) -> str:
    return f"""
    SELECT
        PARSE_DATE('%Y%m%d', SUBSTR(CAST(DATE AS STRING), 1, 8)) AS date,
        COUNT(*) AS ag_volume
    FROM
        `gdelt-bq.gdeltv2.gkg_partitioned`
    WHERE
        _PARTITIONTIME BETWEEN TIMESTAMP('{start_date}') AND TIMESTAMP('{end_date}')
        AND PARSE_DATE('%Y%m%d', SUBSTR(CAST(DATE AS STRING), 1, 8))
            BETWEEN '{start_date}' AND '{end_date}'
        AND (
            V2Themes LIKE '%AGRICULTURE%'
            OR V2Themes LIKE '%FOOD%'
            OR V2Themes LIKE '%ECON_COMMODITY%'
        )
    GROUP BY date
    ORDER BY date
    """  # noqa: S608


def total_news_query(start_date: str, end_date: str) -> str:
    return f"""
    SELECT
        PARSE_DATE('%Y%m%d', SUBSTR(CAST(DATE AS STRING), 1, 8)) AS date,
        COUNT(*) AS total_volume
    FROM
        `gdelt-bq.gdeltv2.gkg_partitioned`
    WHERE
        _PARTITIONTIME BETWEEN TIMESTAMP('{start_date}') AND TIMESTAMP('{end_date}')
        AND PARSE_DATE('%Y%m%d', SUBSTR(CAST(DATE AS STRING), 1, 8))
            BETWEEN '{start_date}' AND '{end_date}'
    GROUP BY date
    ORDER BY date
    """  # noqa: S608
