# Directive: Extract and Clean Environmental and Occupancy Data

## 1. Goal
Extract raw IoT telemetry from the NEMo cloud environmental sensors and building occupancy sensors for **Monash Building 69, Room G.57 (Living Lab 1 & 2 / Smart Infrastructure Lab)**, clean and parse the nested measurement records, align timestamps onto a uniform 5-minute sampling interval, and export an analytical intermediate dataset.

## 2. Inputs
- `Assignment Details/Datasets/5EnvSensor_MayToDec2024_180kRows/5EnvSensor_MayToDec2024_180kRows.csv`: Raw environmental sensor stream.
- `Assignment Details/Datasets/5occupancySensor_MayToDec2024_9MRows/5occupancySensor_MayToDec2024_9MRows.csv`: Raw occupancy event stream.
- `Assignment Details/Datasets/Sensor ID and Locations.xlsx`: Sensor spatial metadata mapping Building 69 G.57 sensors:
  - Environmental Sensor IDs: `6012003000592` (Living Lab 1, ES2G08), `6012003000608` (Living Lab 2, ES2G09).
  - Occupancy Space ID: `9f902aab-58fe-4b1c-ab24-241310df1550` (devices `8034-285a-217d`, `8034-2812-6eb1`).

## 3. Tools & Execution Scripts
- `execution/data_pipeline.py`: Pure deterministic Python script using chunked streaming:
  - Streams `5EnvSensor` CSV, parses JSON payload (`Carbon dioxide`, `Temperature`, `Humidity`, `Particulate matter 2.5`, `Particulate matter 10`, `Formaldehyde`, `VOC`, `Pressure`), filters for sensor IDs `6012003000592` and `6012003000608`.
  - Streams `5occupancySensor` CSV, filters for `floorspaceid == '9f902aab-58fe-4b1c-ab24-241310df1550'`, extracts `headcount` and `occupancystatus`.
  - Resamples both series to 5-minute intervals ($\Delta t = 300\text{ s}$) using median aggregation to mitigate transient outliers.
  - Merges environmental and occupancy streams on UTC timestamps.

## 4. Outputs & Deliverables
- Intermediate: `.tmp/g57_merged_data.parquet` (clean, compact tabular dataset).
- Intermediate summary: `.tmp/data_summary.json` (data date ranges, record counts, null statistics).

## 5. Workflow / Step-by-Step Instructions
1. Run `python execution/data_pipeline.py --env-csv "Assignment Details/Datasets/5EnvSensor_MayToDec2024_180kRows/5EnvSensor_MayToDec2024_180kRows.csv" --occ-csv "Assignment Details/Datasets/5occupancySensor_MayToDec2024_9MRows/5occupancySensor_MayToDec2024_9MRows.csv" --output ".tmp/g57_merged_data.parquet"`
2. Verify `.tmp/g57_merged_data.parquet` exists and non-empty.
3. Validate timestamp continuity and record counts in `.tmp/data_summary.json`.

## 6. Edge Cases & Learnings
- Large file handling: `5occupancySensor` is ~1.6 GB. Do not read the entire CSV into memory with default `pd.read_csv()`. Use chunked reading (`chunksize=100_000`) filtering rows on the fly.
- Sensor dropouts: If a sensor temporarily drops out for $< 30 \text{ min}$, flag for cubic spline or linear interpolation in Directive 02. If dropped out for $> 2 \text{ hours}$, mark period as unavailable.
- Negative / impossible sensor readings: $\text{CO}_2 < 350 \text{ ppm}$ or $\text{Temp} < 10^\circ\text{C}$ indicates sensor calibration faults and should be set to NaN before interpolation.
