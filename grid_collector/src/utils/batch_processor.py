import pandas as pd
import glob
import json
import logging
from datetime import datetime
import os
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def combine_batch_files(pattern: str = 'player_stats_batch_*.csv', archive_batches: bool = True) -> pd.DataFrame:
    """
    Combine all batch CSV files into a single DataFrame and optionally archive batch files.
    
    Args:
        pattern: Glob pattern to match batch files
        archive_batches: Whether to move batch files to dated archive folder
        
    Returns:
        Combined DataFrame
    """
    # Get list of all batch files
    batch_files = glob.glob(pattern)
    logger.info(f"Found {len(batch_files)} batch files to process")
    
    if not batch_files:
        logger.warning("No batch files found!")
        return pd.DataFrame()
    
    all_data = []
    processed_files = []  # Keep track of successfully processed files
    
    for file in batch_files:
        try:
            # Read the CSV file
            df = pd.read_csv(file)
            logger.info(f"Processing {file}: {len(df)} rows")
            
            # Process each row
            for _, row in df.iterrows():
                try:
                    # Convert string representations back to dictionaries where needed
                    processed_row = {}
                    for col, val in row.items():
                        if isinstance(val, str) and (val.startswith('{') or val.startswith('[')):
                            try:
                                processed_row[col] = json.loads(val)
                            except json.JSONDecodeError:
                                processed_row[col] = val
                        else:
                            processed_row[col] = val
                    
                    all_data.append(processed_row)
                    
                except Exception as e:
                    logger.warning(f"Error processing row in {file}: {str(e)}")
                    # Add the raw row to preserve the data
                    all_data.append(row.to_dict())
                    continue
            
            processed_files.append(file)
            
        except Exception as e:
            logger.error(f"Error processing file {file}: {str(e)}")
            continue
    
    if all_data:
        # Create DataFrame from all collected data
        final_df = pd.DataFrame(all_data)
        
        # Save the combined file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        output_file = f'combined_player_stats_{timestamp}.csv'
        final_df.to_csv(output_file, index=False)
        
        # Archive batch files if requested
        if archive_batches and processed_files:
            # Create archive directory with timestamp
            archive_dir = os.path.join('batches', f'batch_archive_{timestamp}')
            os.makedirs(archive_dir, exist_ok=True)
            
            # Move processed files to archive
            for file in processed_files:
                try:
                    shutil.move(file, os.path.join(archive_dir, os.path.basename(file)))
                except Exception as e:
                    logger.error(f"Error moving file {file} to archive: {str(e)}")
            
            logger.info(f"Archived {len(processed_files)} batch files to {archive_dir}")
        
        logger.info(f"Successfully combined {len(final_df)} records into {output_file}")
        
        # Print summary
        print(f"\nCombined Data Summary:")
        print(f"Total records: {len(final_df)}")
        print(f"Columns: {final_df.columns.tolist()}")
        print(f"\nSaved to: {output_file}")
        if archive_batches:
            print(f"Batch files archived to: {archive_dir}")
        
        return final_df
    else:
        logger.warning("No data was successfully processed")
        return pd.DataFrame()

if __name__ == "__main__":
    combined_df = combine_batch_files(archive_batches=True)