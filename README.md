# Content Gap Analysis Tool

A Python tool for analyzing content gaps across competitor websites. This tool processes SEMrush position data and generates comprehensive gap analysis reports.

## Features

- Fast CSV data processing and output
- Processes multiple competitor CSV files simultaneously
- Calculates normalized scores and rankings
- Memory-efficient processing for large datasets
- Comprehensive logging system
- Progress tracking with detailed timing
- Outputs both full and truncated (100k rows) datasets

## Requirements

- Python 3.8+
- pandas
- numpy
- tqdm
- See `requirements.txt` for complete list

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd content-gap-analysis
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install requirements:
```bash
pip install -r requirements.txt
```

## Directory Structure

```
content-gap-analysis/
├── inputs/            # Place input CSV files here
├── logs/             # Log files directory
├── app.py            # Main application
├── requirements.txt  # Python dependencies
└── README.md        # This file
```

## Usage

1. Prepare Input Files:
   - Create an `inputs` folder in the project directory
   - Place your SEMrush position CSVs in the `inputs` folder

2. Configure the Script:
   - Open `app.py`
   - Update the `CLIENT_DOMAIN` variable to match your client's domain

3. Run the Analysis:
```bash
python app.py
```

## Output Files

The script generates two CSV files:
1. `gap_analysis_results.csv` - Top 100,000 rows (truncated version)
2. `gap_analysis_results_full.csv` - Complete dataset (all rows)

## Output Columns

- Keyword
- Search Volume
- Competitor Rankings
- Normalized Scores
- Combined Score
- Additional SEMrush Metrics
- Plus various calculated fields

## Logging

- Logs are saved in the `logs` directory
- Each run creates a new timestamped log file
- Logs include detailed timing and processing information

## Performance

- Memory-efficient processing
- Progress bars for long-running operations
- Automatic garbage collection
- CSV output for faster processing

## Error Handling

- Comprehensive error logging
- Graceful failure handling
- Detailed error messages in logs
- Memory cleanup on errors

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License

## Support

For issues and feature requests, please create an issue in the repository.