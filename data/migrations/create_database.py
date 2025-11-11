from sqlalchemy import create_engine, text, inspect
import pandas as pd
import os
from pathlib import Path
import time
import sys
import gc
import concurrent.futures
import math
import psycopg2
import numpy as np


# Make the directory the script is in the working directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# Get DATABASE_URL from environment (Heroku provides this) or use local config
DATABASE_URL = os.environ.get('DATABASE_URL')

if DATABASE_URL:
    # Heroku's DATABASE_URL starts with postgres://, but SQLAlchemy requires postgresql://
    if DATABASE_URL.startswith('postgres://'):
        DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)
    db_url = DATABASE_URL
    print("Using Heroku PostgreSQL database for migrations")
else:
    # Local database fallback
    db_url = "postgresql+psycopg2://postgres:isoforms@localhost:5432/ad_dash_app"
    print("Using local PostgreSQL database for migrations")

engine = create_engine(
    db_url, 
    pool_size=20,               # Increased for parallel operations
    max_overflow=40,            # Allow more overflow connections
    pool_timeout=60,            # Increased timeout for long operations
    pool_pre_ping=True,         # Check connection validity before using
    connect_args={              # Optimize PostgreSQL client settings
        "application_name": "database_creator",
        "client_encoding": "UTF8",
        "options": "-c statement_timeout=0 -c work_mem=1GB -c maintenance_work_mem=2097151kB"
    }
)

# Configuration options - adjust based on your system
BATCH_SIZE = 1000000           # Number of rows per batch
MAX_WORKERS = 8                # Adjust based on CPU core count
PARALLEL_TABLES = 4            # Number of tables to process in parallel
SAMPLE_SIZE = 100000           # Size of each sample for column type analysis
NUM_SAMPLES = 10               # Number of samples to take throughout the file

# Temporary placeholder for backward compatibility
duck_conn = None

def optimize_postgresql_settings():
    """Apply optimal PostgreSQL settings for the database creation task."""
    try:
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            # Only set these for the current session
            settings = [
                # Use more memory for sorts
                "SET work_mem = '1GB'",
                # Use more memory for maintenance operations - max allowed is 2097151kB
                "SET maintenance_work_mem = '2097151kB'",
                # Disable cost-based vacuum delay
                "SET vacuum_cost_delay = 0",
                # Disable autovacuum during this operation
                "SET autovacuum = off",
                # Set checkpoint segments higher to reduce writes
                "SET checkpoint_timeout = '15min'",
                # Optimize for bulk operations
                "SET synchronous_commit = off"
            ]
            
            for setting in settings:
                conn.execute(text(setting))
                
            print("Successfully applied optimized PostgreSQL settings")
            return True
    except Exception as e:
        print(f"Could not optimize PostgreSQL settings: {e}")
        print("Continuing with default settings")
        return False

def sample_file_for_column_ranges(file_path, sep="\t", dtype=None, num_samples=NUM_SAMPLES, sample_size=SAMPLE_SIZE):
    """
    Sample a large file at different positions to get a more accurate range of values
    for determining column types.
    """
    print(f"Analyzing value ranges across the file: {file_path}")
    start_time = time.time()
    
    # Get file size
    file_size = os.path.getsize(file_path)
    
    # We'll always read the first chunk
    samples = [0]
    
    # Add positions throughout the file to sample
    if file_size > BATCH_SIZE:
        # Calculate sample positions
        positions = [int(p * file_size) for p in np.linspace(0.1, 0.9, num_samples-1)]
        samples.extend(positions)
    
    # Initialize dictionaries to track min/max for integer and float columns
    int_min_values = {}
    int_max_values = {}
    float_min_values = {}
    float_max_values = {}
    float_precision_needed = {}
    column_types = {}
    
    # Sample the file at different positions
    for i, pos in enumerate(samples):
        try:
            # Seek to position (for non-zero positions)
            if pos > 0:
                # Skip to approximate position and read to next line break
                with open(file_path, 'r', encoding='utf-8') as f:
                    f.seek(pos)
                    # Skip current line which might be partial
                    f.readline()
                    # Get position after skipping the line
                    pos = f.tell()
            
            # Read a sample from this position
            sample_df = pd.read_csv(
                file_path, 
                sep=sep, 
                nrows=sample_size, 
                skiprows=0 if pos == 0 else None,
                header=0 if pos == 0 or i == 0 else None,
                names=None if pos == 0 or i == 0 else column_types.keys(),
                dtype=dtype,
                low_memory=False,
                skipfooter=0,
                engine='c',
                skipinitialspace=True,
                skip_blank_lines=True
            )
            
            # If this is the first chunk, record the column names and initialize tracking
            if i == 0:
                for col in sample_df.columns:
                    column_types[col] = sample_df[col].dtype
                    
                    if pd.api.types.is_integer_dtype(sample_df[col].dtype):
                        non_null = sample_df[col].dropna()
                        if len(non_null) > 0:
                            int_min_values[col] = non_null.min()
                            int_max_values[col] = non_null.max()
                    
                    elif pd.api.types.is_float_dtype(sample_df[col].dtype):
                        non_null = sample_df[col].dropna()
                        if len(non_null) > 0:
                            float_min_values[col] = non_null.min()
                            float_max_values[col] = non_null.max()
                            
                            # Check decimal precision
                            max_decimal_places = 0
                            sample_values = non_null.sample(min(100, len(non_null)))
                            for val in sample_values:
                                str_val = str(val)
                                if '.' in str_val:
                                    decimal_places = len(str_val.split('.')[1].rstrip('0'))
                                    max_decimal_places = max(max_decimal_places, decimal_places)
                            float_precision_needed[col] = max_decimal_places
            else:
                # For subsequent chunks, update min/max values
                for col in sample_df.columns:
                    if pd.api.types.is_integer_dtype(sample_df[col].dtype):
                        non_null = sample_df[col].dropna()
                        if len(non_null) > 0:
                            if col in int_min_values:
                                int_min_values[col] = min(int_min_values[col], non_null.min())
                            else:
                                int_min_values[col] = non_null.min()
                                
                            if col in int_max_values:
                                int_max_values[col] = max(int_max_values[col], non_null.max())
                            else:
                                int_max_values[col] = non_null.max()
                    
                    elif pd.api.types.is_float_dtype(sample_df[col].dtype):
                        non_null = sample_df[col].dropna()
                        if len(non_null) > 0:
                            if col in float_min_values:
                                float_min_values[col] = min(float_min_values[col], non_null.min())
                            else:
                                float_min_values[col] = non_null.min()
                                
                            if col in float_max_values:
                                float_max_values[col] = max(float_max_values[col], non_null.max())
                            else:
                                float_max_values[col] = non_null.max()
                            
                            # Update decimal precision
                            max_decimal_places = 0
                            sample_values = non_null.sample(min(100, len(non_null)))
                            for val in sample_values:
                                str_val = str(val)
                                if '.' in str_val:
                                    decimal_places = len(str_val.split('.')[1].rstrip('0'))
                                    max_decimal_places = max(max_decimal_places, decimal_places)
                                    
                            if col in float_precision_needed:
                                float_precision_needed[col] = max(float_precision_needed[col], max_decimal_places)
                            else:
                                float_precision_needed[col] = max_decimal_places
            
            # Clear memory
            del sample_df
            gc.collect()
            
        except Exception as e:
            print(f"Warning: Error sampling file at position {pos}: {e}")
            continue
    
    # Determine optimal types based on the sampled data
    optimal_types = {}
    
    # Integer ranges for PostgreSQL data types
    smallint_min, smallint_max = -32768, 32767
    integer_min, integer_max = -2147483648, 2147483647
    
    # Process integer columns
    for col in int_min_values.keys():
        min_val = int_min_values[col]
        max_val = int_max_values[col]
        
        # Conservative approach - prefer INTEGER over SMALLINT to be safe
        if min_val >= smallint_min and max_val <= smallint_max and min_val > -10000 and max_val < 10000:
            # Only use SMALLINT if the values are well within the range (safety margin)
            optimal_types[col] = 'SMALLINT'
        elif min_val >= integer_min and max_val <= integer_max:
            optimal_types[col] = 'INTEGER'
        else:
            optimal_types[col] = 'BIGINT'  # Default to BIGINT for large values
    
    # Process float columns
    for col in float_min_values.keys():
        min_val = float_min_values[col]
        max_val = float_max_values[col]
        precision = float_precision_needed.get(col, 0)
        
        # Very conservative approach for REAL
        if precision <= 6 and abs(min_val) < 1e30 and abs(max_val) < 1e30:
            optimal_types[col] = 'REAL'
        else:
            optimal_types[col] = 'DOUBLE PRECISION'
    
    elapsed_time = time.time() - start_time
    print(f"Completed column type analysis in {elapsed_time:.2f} seconds")
    
    return optimal_types

def determine_column_types(df):
    """Determine optimal column types based on data values in a DataFrame."""
    optimized_types = {}
    
    # Integer ranges for PostgreSQL data types
    smallint_min, smallint_max = -32768, 32767
    integer_min, integer_max = -2147483648, 2147483647
    
    for column in df.columns:
        if pd.api.types.is_integer_dtype(df[column].dtype):
            # Check range of values to determine appropriate integer type
            if df[column].isna().all():
                # All values are NaN, use default
                continue
                
            non_null = df[column].dropna()
            if len(non_null) == 0:
                continue
                
            min_val = non_null.min()
            max_val = non_null.max()
            
            # Conservative approach - prefer INTEGER over SMALLINT
            if min_val >= smallint_min and max_val <= smallint_max and min_val > -10000 and max_val < 10000:
                # Only use SMALLINT if well within range with safety margin
                optimized_types[column] = 'SMALLINT'
            elif min_val >= integer_min and max_val <= integer_max:
                optimized_types[column] = 'INTEGER'
            # Otherwise keep as BIGINT (default)
            
        elif pd.api.types.is_float_dtype(df[column].dtype):
            # Check if DOUBLE PRECISION can be converted to REAL
            non_null = df[column].dropna()
            if len(non_null) == 0:
                continue
                
            # Check for very large or very small values that might need double precision
            max_abs_value = non_null.abs().max()
            if max_abs_value <= 3.4e38 and max_abs_value >= 1.17e-38:
                # Check decimal precision
                # Convert to string and find max decimal places needed
                decimal_places = 0
                sample = non_null.sample(min(1000, len(non_null)))
                
                for val in sample:
                    str_val = str(val)
                    if '.' in str_val:
                        decimal_part = str_val.split('.')[1].rstrip('0')
                        decimal_places = max(decimal_places, len(decimal_part))
                
                # REAL has ~6-7 significant decimal digits of precision
                if decimal_places <= 6:
                    optimized_types[column] = 'REAL'
            # Otherwise keep as DOUBLE PRECISION (default)
    
    return optimized_types

def process_file(file_path):
    """Process a single file and create a database table."""
    try:
        print(f"Loading file: {file_path}")
        start_time = time.time()
        
        # Create table name from file name without extension
        table_name = file_path.stem.lower()
        
        # For transcript_annotation.tsv, we need to handle the seqnames column differently
        if table_name == "transcript_annotation":
            ## Define dtypes for the table
            dtypes = {'gene_id': str, 'gene_name': str, 'transcript_id': str, 'transcript_name': str, 
                     'transcript_biotype': str, 'seqnames': str, 'strand': str, 'type': str, 
                     'start': int, 'end': int, 'exon_number': int}
            
            # First analyze the file to determine column types
            optimized_types = sample_file_for_column_ranges(
                file_path, 
                sep="\t", 
                dtype=dtypes, 
                num_samples=NUM_SAMPLES,
                sample_size=SAMPLE_SIZE
            )
            print(f"Optimized column types for {table_name}: {optimized_types}")
            
            # Force specific columns to INTEGER type
            for col in ["gene_index", "transcript_index", "rsid_index", "start", "end", "pos"]:
                if col in optimized_types:
                    optimized_types[col] = "INTEGER"
                    print(f"Forcing {col} column to INTEGER as requested")
            
            # Read and process file in chunks
            chunk_size = BATCH_SIZE
            first_chunk = True
            
            for i, chunk in enumerate(pd.read_csv(file_path, sep="\t", chunksize=chunk_size, dtype=dtypes)):
                if first_chunk:
                    # Create table with first chunk
                    with engine.begin() as conn:
                        # First just create the table with pandas
                        chunk.to_sql(table_name, conn, if_exists='replace', index=False)
                        
                        # Then alter column types to optimized versions
                        for column, column_type in optimized_types.items():
                            try:
                                conn.execute(text(f'ALTER TABLE "{table_name}" ALTER COLUMN "{column}" TYPE {column_type}'))
                                print(f"Optimized column {column} to {column_type}")
                            except Exception as e:
                                print(f"Could not optimize column {column}: {e}")
                    
                    first_chunk = False
                    print(f"Created table '{table_name}' with optimized data types")
                else:
                    # For subsequent chunks, convert data types before insertion if needed
                    for column, column_type in optimized_types.items():
                        if column in chunk.columns:
                            # Convert the column to appropriate type if needed
                            if column_type == 'SMALLINT' or column_type == 'INTEGER':
                                chunk[column] = chunk[column].astype('int32')
                            elif column_type == 'REAL':
                                chunk[column] = chunk[column].astype('float32')
                    
                    # Append to table
                    chunk.to_sql(table_name, engine, if_exists='append', index=False)
                    print(f"Appended chunk {i+1} to table '{table_name}'")
                
                # Clear memory
                del chunk
                gc.collect()
        else:
            # For other files, use the optimized approach
            # First analyze the file to determine column types
            optimized_types = sample_file_for_column_ranges(
                file_path, 
                sep="\t", 
                num_samples=NUM_SAMPLES,
                sample_size=SAMPLE_SIZE
            )
            print(f"Optimized column types for {table_name}: {optimized_types}")
            
            # Hard-code the average_coverage column on the coverage table as INTEGER
            if table_name == "coverage_table" and "average_coverage" in optimized_types:
                optimized_types["average_coverage"] = "INTEGER"
                print("Forcing average_coverage column to INTEGER as requested")

            # === Add specific handling for genotyping table ===
            if table_name == "genotyping":
                print(f"Applying specific type overrides for genotyping table...")
                new_optimized_types = {}
                for column, dtype in optimized_types.items():
                    if column == "rsid_index":
                        new_optimized_types[column] = "INTEGER"
                    # Check if the sampled type is an integer variant
                    elif dtype in ['SMALLINT', 'INTEGER', 'BIGINT']:
                        new_optimized_types[column] = "SMALLINT"
                    else:
                        # Keep non-integer types as determined by sampling
                        new_optimized_types[column] = dtype
                optimized_types = new_optimized_types
                print(f"Final optimized types for genotyping: {optimized_types}")
            # === End specific handling ===

            # Force specific columns to INTEGER type regardless of table
            # Note: We remove rsid_index here as it's handled above for genotyping specifically
            # and the generic logic below might conflict or be redundant.
            # For other tables, rsid_index will still be handled below if present.
            force_integer_cols = ["gene_index", "transcript_index", "start", "end", "pos"]
            if table_name != "genotyping" and "rsid_index" in optimized_types:
                 force_integer_cols.append("rsid_index") # Add it back for non-genotyping tables

            for col in force_integer_cols:
                if col in optimized_types:
                    optimized_types[col] = "INTEGER"
                    print(f"Forcing {col} column to INTEGER as requested")
            
            # Read and process file in chunks
            chunk_size = BATCH_SIZE
            first_chunk = True
            
            for i, chunk in enumerate(pd.read_csv(file_path, sep="\t", chunksize=chunk_size, low_memory=False)):
                if first_chunk:
                    # Create table with first chunk
                    with engine.begin() as conn:
                        # First just create the table with pandas
                        chunk.to_sql(table_name, conn, if_exists='replace', index=False)
                        
                        # Then alter column types to optimized versions
                        for column, column_type in optimized_types.items():
                            try:
                                conn.execute(text(f'ALTER TABLE "{table_name}" ALTER COLUMN "{column}" TYPE {column_type}'))
                                print(f"Optimized column {column} to {column_type}")
                            except Exception as e:
                                print(f"Could not optimize column {column}: {e}")
                    
                    first_chunk = False
                    print(f"Created table '{table_name}' with optimized data types")
                else:
                    # For subsequent chunks, convert data types before insertion if needed
                    for column, column_type in optimized_types.items():
                        if column in chunk.columns:
                            # Convert the column to appropriate type if needed
                            if column_type == 'SMALLINT' or column_type == 'INTEGER':
                                chunk[column] = chunk[column].astype('int32')
                            elif column_type == 'REAL':
                                chunk[column] = chunk[column].astype('float32')
                    
                    # Append to table
                    chunk.to_sql(table_name, engine, if_exists='append', index=False)
                    print(f"Appended chunk {i+1} to table '{table_name}'")
                
                # Clear memory
                del chunk
                gc.collect()
        
        print(f"Successfully loaded all data into table '{table_name}'")
        
        # Create indexes with optimized settings
        create_indexes(table_name)
        
        elapsed_time = time.time() - start_time
        print(f"Processed {table_name} in {elapsed_time:.2f} seconds")
        
        return True
    except Exception as e:
        print(f"Failed to load {file_path}: {e}")
        return False

def create_indexes(table_name):
    """Create optimized indexes for the table."""
    try:
        # Identify columns to index based on specific requirements
        columns_to_index = []
        
        # Get column names without loading entire dataframe
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table_name}'"))
            columns = [row[0] for row in result.fetchall()]
        
        # Check for gene_id and gene_name
        if "gene_id" in columns and "gene_name" in columns:
            columns_to_index.append("gene_id")
            columns_to_index.append("gene_name")

        # Check for gene_id if gene_name does not exist
        if "gene_id" in columns and "gene_name" not in columns:
            columns_to_index.append("gene_id")

        # Check for sex if sample_id does not exist
        if "sex" in columns and "sample_id" not in columns:
            columns_to_index.append("sex")
        
        # Check for independent_variable if sample_id does not exist
        if "independent_variable" in columns and "sample_id" not in columns:
            columns_to_index.append("independent_variable")
        
        # Check for gene_index if gene_id does not exist
        if "gene_id" not in columns and "gene_index" in columns:
            columns_to_index.append("gene_index")

        # Check for rsid
        if "rsid" in columns:
            columns_to_index.append("rsid")

        # Check for rsid
        if "rsid" not in columns and "rsid_index" in columns:
            columns_to_index.append("rsid_index")

        if "pos" in columns and "seqnames_index" in columns and "strand" in columns:
            columns_to_index.append("pos")
            columns_to_index.append("seqnames_index")
            columns_to_index.append("strand")

        if "genotype_index" in columns:
            columns_to_index.append("genotype_index")
        
        # Set optimal maintenance_work_mem for index creation
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            conn.execute(text("SET maintenance_work_mem = '2097151kB'"))
        
        # Add indexes on the identified columns
        if columns_to_index:
            for col in columns_to_index:
                try:
                    # Create a new connection for each index creation to avoid transaction blocks
                    # CREATE INDEX CONCURRENTLY cannot run inside a transaction block
                    index_name = f"idx_{table_name}_{col}"
                    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
                        print(f"Creating index on column '{col}' for table '{table_name}'")
                        conn.execute(text(f'CREATE INDEX CONCURRENTLY {index_name} ON "{table_name}" ("{col}");'))
                        
                        # Verify the index was created
                        index_check_query = text(f"""
                        SELECT indexname 
                        FROM pg_indexes 
                        WHERE tablename = '{table_name}' 
                        AND indexname = '{index_name}'
                        """)
                        
                        result = conn.execute(index_check_query)
                        index_exists = result.fetchone()
                        
                        if index_exists:
                            print(f"Verified index '{index_name}' was successfully created")
                        else:
                            print(f"WARNING: Index '{index_name}' was not found after creation attempt")
                            
                except Exception as e:
                    print(f"Failed to create index on column '{col}': {e}")
        else:
            print(f"No indexes created for table '{table_name}' as it doesn't contain gene_id, gene_name, or rsid columns")
        
        return True
    except Exception as e:
        print(f"Error creating indexes for {table_name}: {e}")
        return False

def main():
    print("Starting database creation with optimized settings...")
    start_time = time.time()
    
    # Apply optimized PostgreSQL settings
    optimize_postgresql_settings()
    
    try:
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as connection:
            # This query returns the name of the current database.
            result = connection.execute(text("SELECT current_database();"))
            db_name = result.fetchone()[0]
            print(f"Successfully connected to the database: {db_name}")
            
            # Drop all existing tables to start with a clean slate
            connection.execute(text("DROP SCHEMA public CASCADE;"))
            connection.execute(text("CREATE SCHEMA public;"))
            connection.execute(text("GRANT ALL ON SCHEMA public TO postgres;"))
            connection.execute(text("GRANT ALL ON SCHEMA public TO public;"))
            
            # Verify that tables were actually cleared
            result = connection.execute(text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """))
            remaining_tables = result.fetchall()
            
            if not remaining_tables:
                print("Confirmed: All existing tables have been cleared from the database")
            else:
                table_names = [table[0] for table in remaining_tables]
                print(f"Warning: Some tables still exist: {', '.join(table_names)}")
    except Exception as e:
        print("Failed to connect to the database or clear tables:", e)
        return

    # Open files in ../raw_data/ using pandas
    raw_data_dir = Path("../raw_data/")
    processed_count = 0

    if raw_data_dir.exists():
        # Get list of all files
        files = [f for f in raw_data_dir.glob("*") if f.is_file()]
        
        if len(files) > 0:
            print(f"Found {len(files)} files to process")
            
            # Process files in parallel if we have multiple files
            if len(files) > 1 and len(files) <= PARALLEL_TABLES:
                print(f"Processing {len(files)} files in parallel with {min(len(files), MAX_WORKERS)} workers")
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(files), MAX_WORKERS)) as executor:
                    results = list(executor.map(process_file, files))
                    processed_count = sum(1 for result in results if result)
            elif len(files) > PARALLEL_TABLES:
                print(f"Processing {len(files)} files in batches of {PARALLEL_TABLES} in parallel")
                # Process files in batches to avoid memory issues
                for i in range(0, len(files), PARALLEL_TABLES):
                    batch_files = files[i:i+PARALLEL_TABLES]
                    print(f"Processing batch {i//PARALLEL_TABLES + 1} with {len(batch_files)} files")
                    
                    with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch_files)) as executor:
                        results = list(executor.map(process_file, batch_files))
                        processed_count += sum(1 for result in results if result)
                    
                    # Give the system a break between batches
                    gc.collect()
                    time.sleep(2)
            else:
                # Process files sequentially for small number of files
                for file_path in files:
                    if process_file(file_path):
                        processed_count += 1
            
            print(f"Loaded {processed_count} files into database tables with indexes")
            
            # Analyze the database after loading all data
            try:
                with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
                    print("Analyzing database to update statistics...")
                    conn.execute(text("ANALYZE"))
                    
                    # Reset session parameters
                    conn.execute(text("RESET ALL"))
                    
                    # Get database size
                    size_query = text("SELECT pg_size_pretty(pg_database_size(current_database())) as db_size")
                    result = conn.execute(size_query)
                    db_size = result.scalar()
                    print(f"Final database size: {db_size}")
            except Exception as e:
                print(f"Failed to analyze database: {e}")
        else:
            print("No files found in raw_data directory")
    else:
        print("Raw data directory not found at ../raw_data/")
    
    total_time = time.time() - start_time
    print(f"Database creation complete in {total_time:.2f} seconds")

if __name__ == "__main__":
    main()