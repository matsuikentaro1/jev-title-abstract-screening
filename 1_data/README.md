# API input specification

This folder documents the input expected by api_runners/. The input file is not distributed; see the [package README](../README.md) for the sharing scope. Saved-output reanalysis reads data/ and does not use this folder.

The runners expect records.json as a UTF-8 JSON array with these fields:

| Field | Format |
|---|---|
| id | Identifier matching record_id in data/records.csv |
| dataset | takeshima or maruki |
| title | String |
| abstract | String; empty when missing |
