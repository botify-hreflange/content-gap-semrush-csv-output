import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging
from tqdm import tqdm
import time
import sys
import gc

def normalize_domain(domain):
    """Normalize domain name by removing subdomains and www."""
    try:
        if pd.isna(domain):
            return None
            
        # Remove scheme if present (http://, https://)
        if '://' in domain:
            domain = domain.split('://')[1]
        
        # Remove path and query parameters
        domain = domain.split('/')[0]
        
        # Remove www. and other subdomains
        parts = domain.split('.')
        if len(parts) > 2:
            # Keep only the last two parts for .com domains
            domain = '.'.join(parts[-2:])
            
        # Handle special cases like .co.uk
        special_tlds = ['.co.uk', '.com.au', '.co.jp']
        for tld in special_tlds:
            if domain.endswith(tld):
                parts = domain.split('.')
                if len(parts) > 3:
                    domain = '.'.join(parts[-3:])
                break
                
        return domain.lower()
    except:
        return None

def extract_domain_from_url(url):
    """Extract the base domain from a URL."""
    try:
        if pd.isna(url):
            return None
        # Remove http:// or https:// if present
        if '://' in url:
            url = url.split('://')[1]
        # Get the domain part
        domain = url.split('/')[0]
        # Remove www. if present
        if domain.startswith('www.'):
            domain = domain[4:]
        return domain
    except:
        return None

def setup_logging():
    """Set up logging configuration."""
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = logs_dir / f'gap_analysis_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def normalize_score(series, ascending=True):
    """Normalize values to 0-1 range."""
    try:
        if series.empty or series.isna().all():
            return pd.Series(0, index=series.index)
        
        # Convert values to numeric, handling both string and numeric inputs
        if series.dtype == object:
            # Try to convert string numbers with commas to float
            numeric_series = pd.to_numeric(
                series.astype(str).str.replace(',', ''),
                errors='coerce'
            )
        else:
            numeric_series = pd.to_numeric(series, errors='coerce')
        
        # Handle any remaining non-numeric values
        numeric_series = numeric_series.fillna(0)
        
        min_val = numeric_series.min()
        max_val = numeric_series.max()
        
        if min_val == max_val:
            return pd.Series(0, index=series.index)
        
        if ascending:
            return (max_val - numeric_series) / (max_val - min_val)
        else:
            return (numeric_series - min_val) / (max_val - min_val)
    except Exception as e:
        logging.error(f"Error in normalize_score: {str(e)}")
        return pd.Series(0, index=series.index)

def process_competitor_files(dfs, competitor_cols):
    """Process competitor files and extract positions."""
    logger = logging.getLogger(__name__)
    
    # Create a master dataframe to store all keyword-domain-position combinations
    master_data = []
    
    # First collect all unique keywords and their metadata
    keyword_metadata = {}
    
    # Process each file
    for df in dfs:
        try:
            # Extract and normalize domain
            df['domain'] = df['URL'].apply(extract_domain_from_url).apply(normalize_domain)
            
            # Process each row
            for _, row in df.iterrows():
                keyword = row['Keyword']
                domain = row['domain']
                position = row['Position']
                
                # Store domain-position pairs for each keyword
                master_data.append({
                    'Keyword': keyword,
                    'domain': domain,
                    'Position': position
                })
                
                # Store metadata for each keyword (first occurrence)
                if keyword not in keyword_metadata:
                    keyword_metadata[keyword] = {
                        'Search Volume': row['Search Volume'],
                        'Keyword Difficulty': row['Keyword Difficulty'],
                        'CPC': row['CPC'],
                        'Competition': row['Competition'],
                        'SERP Features by Keyword': row['SERP Features by Keyword'],
                        'Keyword Intents': row['Keyword Intents'],
                        'Position Type': row['Position Type'],
                        'Number of Results': row['Number of Results']
                    }
            
            logger.info(f"Processed {len(df)} rows")
            
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            continue
    
    # Convert to DataFrame
    position_df = pd.DataFrame(master_data)
    
    # Get the minimum position for each keyword-domain combination
    position_df = position_df.groupby(['Keyword', 'domain'])['Position'].min().reset_index()
    
    # Pivot to get competitor columns
    result_df = position_df.pivot(index='Keyword', columns='domain', values='Position')
    
    # Add metadata
    metadata_df = pd.DataFrame.from_dict(keyword_metadata, orient='index')
    result_df = result_df.join(metadata_df)
    
    # Reset index to make Keyword a column
    result_df = result_df.reset_index()
    
    # Ensure all competitor columns exist
    for col in competitor_cols:
        if col not in result_df.columns:
            result_df[col] = np.nan
    
    # Sort by Search Volume descending
    result_df['Search Volume'] = pd.to_numeric(result_df['Search Volume'], errors='coerce')
    result_df = result_df.sort_values('Search Volume', ascending=False)
    
    return result_df

def get_competitor_domains(dfs, client_domain):
    """Extract unique competitor domains from input files."""
    logger = logging.getLogger(__name__)
    domains = set()
    
    # Normalize the client domain
    normalized_client_domain = normalize_domain(client_domain)
    
    for df in dfs:
        try:
            # Extract domain from URL column
            df['domain'] = df['URL'].apply(extract_domain_from_url)
            df['domain'] = df['domain'].apply(normalize_domain)
            
            # Add normalized domains to set (including client domain)
            domains.update(df['domain'].dropna().unique())
            
        except Exception as e:
            logger.error(f"Error extracting domains: {str(e)}")
            continue
    
    # Ensure client domain is included even if not in the data
    if normalized_client_domain:
        domains.add(normalized_client_domain)
    
    # Sort domains for consistent ordering
    competitor_cols = sorted(list(domains))
    logger.info(f"Found {len(competitor_cols)} domains (including client)")
    logger.info(f"All domains: {', '.join(competitor_cols)}")
    
    return competitor_cols

def process_gap_analysis(input_folder, output_file, client_domain):
    """Process competitor gap analysis and create CSV output."""
    logger = setup_logging()
    start_time = time.time()
    logger.info(f"Starting gap analysis processing for {client_domain}")
    
    try:
        # Read all CSV files from input folder
        input_path = Path(input_folder)
        dfs = []
        
        csv_files = list(input_path.glob('*.csv'))
        if not csv_files:
            logger.error("No CSV files found in input folder")
            raise ValueError("No CSV files found in input folder")
        
        logger.info(f"Found {len(csv_files)} CSV files to process")
        
        # Read CSV files with progress bar
        for csv_file in tqdm(csv_files, desc="Reading CSV files"):
            try:
                logger.info(f"Processing {csv_file.name}")
                df = pd.read_csv(csv_file)
                dfs.append(df)
                logger.info(f"Successfully read {len(df):,} rows from {csv_file.name}")
            except Exception as e:
                logger.error(f"Error processing {csv_file.name}: {str(e)}")
                continue
        
        # Get competitor columns dynamically from input files (including client)
        logger.info("Identifying all domains")
        competitor_cols = get_competitor_domains(dfs, client_domain)
        
        logger.info("Processing domain data")
        combined_df = process_competitor_files(dfs, competitor_cols)
        logger.info(f"Combined shape: {combined_df.shape}")
        
        # Calculate additional columns
        logger.info("Calculating additional columns")
        combined_df['Number of Words'] = combined_df['Keyword'].str.split().str.len()
        combined_df['Competitors Positioning'] = combined_df[competitor_cols].notna().sum(axis=1)
        
        # Set keyword group to General
        combined_df['Keyword Group (Experimental)'] = 'General'
        
        # Handle Search Volume formatting
        combined_df['Search Volume'] = pd.to_numeric(combined_df['Search Volume'], errors='coerce')
        
        # Normalize scores
        logger.info("Calculating normalized scores")
        score_columns = {
            'Search Volume': False,
            'Keyword Difficulty': True,
            'CPC': False,
            'Competition': False
        }
        
        # Calculate normalized scores
        for col, ascending in score_columns.items():
            if col in combined_df.columns:
                norm_col = f'Normalized {col}'
                combined_df[norm_col] = normalize_score(combined_df[col], ascending)
                logger.info(f"Normalized {col}")
        
        # Calculate combined score
        score_columns = [col for col in combined_df.columns if col.startswith('Normalized')]
        combined_df['Combined Score'] = combined_df[score_columns].mean(axis=1)
        
        # Add timestamp
        combined_df['Timestamp'] = datetime.now().strftime('%-m/%-d/%y')
        
        # Define column order
        output_columns = [
            'Keyword', 'Search Volume',
            *competitor_cols,  # This now includes the client domain
            'Competitors Positioning', 'Number of Words',
            'Keyword Group (Experimental)', 'Keyword Difficulty',
            'CPC', 'Competition', 'SERP Features by Keyword',
            'Keyword Intents', 'Position Type', 'Number of Results',
            'Timestamp',
            *[col for col in combined_df.columns if col.startswith('Normalized')],
            'Combined Score'
        ]
        
        # Sort and save results
        logger.info("Sorting and saving results")
        combined_df = combined_df.sort_values('Search Volume', ascending=False)
        
        # Filter columns that exist in the dataframe
        output_columns = [col for col in output_columns if col in combined_df.columns]
        
        # Save to CSV
        logger.info(f"Saving results to {output_file}")
        combined_df[output_columns].to_csv(output_file, index=False)
        
        logger.info(f"Analysis complete. Results saved to {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error in process_gap_analysis: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    # Configuration
    INPUT_FOLDER = "inputs"
    OUTPUT_FILE = "gap_analysis_results.csv"
    CLIENT_DOMAIN = "kay.com"  # Base domain without www or other subdomains
    
    try:
        start_time = time.time()
        success = process_gap_analysis(INPUT_FOLDER, OUTPUT_FILE, CLIENT_DOMAIN)
        elapsed = time.time() - start_time
        
        if success:
            print(f"\nAnalysis complete in {elapsed:.2f} seconds")
            print(f"Results saved to {OUTPUT_FILE}")
    except Exception as e:
        print(f"\nError: {str(e)}")
        sys.exit(1)