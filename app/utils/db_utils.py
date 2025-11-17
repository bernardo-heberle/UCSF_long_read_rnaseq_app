import pandas as pd
try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    print("Warning: Polars is not available. Using Pandas only.")

from sqlalchemy import create_engine, text
import os
import threading
import time

# Configure polars to not truncate output in print statements and head() calls
pl.Config.set_tbl_rows(100)  # Show up to 100 rows
pl.Config.set_tbl_cols(100)  # Show up to 100 columns
pl.Config.set_tbl_width_chars(2000)  # Increase width for better display
pl.Config.set_fmt_str_lengths(100)  # Don't truncate string values

# Configure pandas to not truncate output in print statements and head() calls
pd.set_option('display.max_rows', 100)  # Show up to 100 rows
pd.set_option('display.max_columns', 100)  # Show up to 100 columns
pd.set_option('display.width', 2000)  # Increase width for better display
pd.set_option('display.max_colwidth', 100)  # Don't truncate string values

# Get DATABASE_URL from environment (Heroku provides this) or use local config
DATABASE_URL = os.environ.get('DATABASE_URL')

# Initialize SQLAlchemy engine - global connection
if DATABASE_URL:
    # Heroku's DATABASE_URL starts with postgres://, but SQLAlchemy requires postgresql://
    if DATABASE_URL.startswith('postgres://'):
        DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)
    pg_engine = create_engine(DATABASE_URL)
    print("Connected to Heroku PostgreSQL database")
else:
    # Local database fallback
    PG_HOST = 'localhost'
    PG_PORT = '5432'
    PG_DB = 'ucsf_rnaseq_app'
    PG_USER = 'postgres'
    PG_PASSWORD = 'isoforms'
    pg_engine = create_engine(f'postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}')
    print("Connected to local PostgreSQL database")

# DuckDB compatibility wrapper around Postgres connection
class PostgresDuckWrapper:
    def __init__(self, engine):
        self.engine = engine
    
    def execute(self, query, params=None):
        """Execute a query with parameters and return a result that mimics DuckDB's interface"""
        with self.engine.connect() as conn:
            # Convert DuckDB-style ? params to SQLAlchemy-style :param format
            if params and '?' in query:
                # Replace ? with :param0, :param1, etc.
                param_count = query.count('?')
                for i in range(param_count):
                    query = query.replace('?', f':param{i}', 1)
                
                # Convert list params to dict params
                if isinstance(params, list):
                    params_dict = {f'param{i}': params[i] for i in range(len(params))}
                    params = params_dict
            
            result = conn.execute(text(query), params)
            
            # Add polars compatibility
            result.pl = lambda: pl.DataFrame(pd.DataFrame(result.fetchall(), columns=result.keys())) if POLARS_AVAILABLE else None
            
            return result

# Create the duck_conn compatibility wrapper
duck_conn = PostgresDuckWrapper(pg_engine)

# Cache dictionaries for commonly accessed data
GENE_INFO_CACHE = {}
MATRIX_DATA_CACHE = {}
SEARCH_RESULTS_CACHE = {}  # Cache for search results
ALL_GENES = []  # Global list to hold all genes
GENE_INDEX_LOADED = False  # Flag to track if the gene index is loaded
GENE_INDEX_LOADING = False  # Flag to track if the gene index is currently loading

def _load_gene_index_thread():
    """Background thread to load gene index"""
    global ALL_GENES, GENE_INDEX_LOADED, GENE_INDEX_LOADING
    
    try:
        # Set the loading flag
        GENE_INDEX_LOADING = True
        
        with pg_engine.connect() as conn:
            query = text("""
                SELECT DISTINCT gene_id, gene_name, gene_index
                FROM gene_and_transcript_index_table
                ORDER BY gene_name
            """)
            all_genes_raw = conn.execute(query).fetchall()
            
            # Store as dictionary for faster lookup
            # Using a dict comprehension to ensure uniqueness by gene_id
            gene_dict = {}
            for gene_id, gene_name, gene_index in all_genes_raw:
                if gene_id not in gene_dict:
                    gene_dict[gene_id] = (gene_name, gene_index)
            
            # Convert to list of tuples for searching
            ALL_GENES = [(gene_id, gene_name, gene_index) for gene_id, (gene_name, gene_index) in gene_dict.items()]
            
            GENE_INDEX_LOADED = True
            
    except Exception as e:
        print(f"Error loading gene index in background: {e}")
        # Set empty list in case of failure
        ALL_GENES = []
    finally:
        GENE_INDEX_LOADING = False
    
def start_async_gene_index_load():
    """Start async loading of gene index in a background thread"""
    global GENE_INDEX_LOADING
    
    if GENE_INDEX_LOADED or GENE_INDEX_LOADING:
        return
        
    # Create and start background thread
    thread = threading.Thread(target=_load_gene_index_thread)
    thread.daemon = True  # Make thread exit when main program exits
    thread.start()
    
# Start loading the gene index in the background
start_async_gene_index_load()

def _load_gene_index():
    """
    Synchronous function to load the gene index if needed.
    Will wait for background loading if it's in progress.
    """
    global GENE_INDEX_LOADED, GENE_INDEX_LOADING
    
    # If already loaded, return immediately
    if GENE_INDEX_LOADED:
        return
    
    # If background loading is in progress, wait for it to complete
    if GENE_INDEX_LOADING:
        print("Gene index loading in progress, waiting for completion...")
        while GENE_INDEX_LOADING:
            time.sleep(0.1)  # Small sleep to reduce CPU usage while waiting
        return
    
    # If not loaded or loading, do a synchronous load
    try:
        print("Loading gene index synchronously...")
        with pg_engine.connect() as conn:
            query = text("""
                SELECT DISTINCT gene_id, gene_name, gene_index
                FROM gene_and_transcript_index_table
                ORDER BY gene_name
            """)
            all_genes_raw = conn.execute(query).fetchall()
            
            # Store as dictionary for faster lookup
            gene_dict = {}
            for gene_id, gene_name, gene_index in all_genes_raw:
                if gene_id not in gene_dict:
                    gene_dict[gene_id] = (gene_name, gene_index)
            
            # Convert to list of tuples for searching
            ALL_GENES = [(gene_id, gene_name, gene_index) for gene_id, (gene_name, gene_index) in gene_dict.items()]
            
            GENE_INDEX_LOADED = True
            print(f"Gene index loaded synchronously: {len(ALL_GENES)} unique genes")
    except Exception as e:
        print(f"Error loading gene index synchronously: {e}")
        # Set empty list in case of failure
        ALL_GENES = []

def get_matrix_dropdown_options():
    """Get formatted dropdown options for different counting methods."""
    # Define options for different counting methods that will use columns from all_transcript_data
    ordered_options = [
        {'label': 'Unique Counts', 'value': 'unique'},
        {'label': 'Total Counts', 'value': 'total'},
        {'label': 'Full Length Counts', 'value': 'fullLength'}
    ]
    
    return ordered_options

def search_genes(search_value, previous_search=None):
    """
    Search for genes using the in-memory index.
    
    Args:
        search_value (str): The current search value
        previous_search (str): The previous search value, not used in this implementation
    """
    if not search_value:
        return []
    
    # Make sure gene index is loaded
    if not GENE_INDEX_LOADED:
        _load_gene_index()
    
    # If still no genes, fall back to database search
    if not ALL_GENES:
        return _search_genes_database(search_value)
    
    # Convert to lowercase once
    search_value = search_value.lower()
    
    # In-memory search
    filtered_results = []
    exact_matches = []
    prefix_matches = []
    contains_matches = []
    
    # First pass: categorize matches
    for gene_id, gene_name, gene_index in ALL_GENES:
        lower_id = gene_id.lower()
        lower_name = gene_name.lower()
        
        # Exact match (highest priority)
        if lower_id == search_value or lower_name == search_value:
            exact_matches.append((gene_id, gene_name, gene_index))
        # Prefix match (medium priority)
        elif lower_id.startswith(search_value) or lower_name.startswith(search_value):
            prefix_matches.append((gene_id, gene_name, gene_index))
        # Contains match (lowest priority)
        elif search_value in lower_id or search_value in lower_name:
            contains_matches.append((gene_id, gene_name, gene_index))
            
        # Stop when we have enough matches
        if len(exact_matches) >= 10:
            break
    
    # Combine results in priority order
    filtered_results = exact_matches
    
    # Add prefix matches if needed
    if len(filtered_results) < 10:
        filtered_results.extend(prefix_matches[:10 - len(filtered_results)])
    
    # Add contains matches if needed
    if len(filtered_results) < 10:
        filtered_results.extend(contains_matches[:10 - len(filtered_results)])
    
    # Convert to options format
    options = []
    for gene_id, gene_name, gene_index in filtered_results[:10]:
        options.append({
            'label': f"{gene_name} ({gene_id})",
            'value': gene_index
        })
    
    return options

def _search_genes_database(search_value):
    """Fallback to database search if in-memory index fails"""
    try:
        with pg_engine.connect() as conn:
            # For very short searches, just do a prefix match
            if len(search_value) < 3:
                query = text("""
                    SELECT DISTINCT gene_id, gene_name, gene_index
                    FROM gene_and_transcript_index_table 
                    WHERE LOWER(gene_id) LIKE :prefix
                    OR LOWER(gene_name) LIKE :prefix
                    LIMIT 10
                """)
                result = conn.execute(query, {"prefix": f"{search_value.lower()}%"}).fetchall()
            else:
                # For longer searches, do a more comprehensive search
                query = text("""
                    WITH ranked_results AS (
                        SELECT DISTINCT 
                            gene_id,
                            gene_name,
                            gene_index,
                            CASE 
                                WHEN LOWER(gene_id) = :exact THEN 1
                                WHEN LOWER(gene_name) = :exact THEN 2
                                WHEN LOWER(gene_id) LIKE :prefix THEN 3
                                WHEN LOWER(gene_name) LIKE :prefix THEN 4
                                ELSE 5
                            END as match_rank
                        FROM gene_and_transcript_index_table 
                        WHERE (
                            LOWER(gene_id) LIKE :pattern
                            OR LOWER(gene_name) LIKE :pattern
                        )
                    )
                    SELECT gene_id, gene_name, gene_index
                    FROM ranked_results
                    ORDER BY match_rank, gene_name
                    LIMIT 10
                """)
                
                result = conn.execute(query, {
                    "exact": search_value.lower(),
                    "prefix": f"{search_value.lower()}%",
                    "pattern": f"%{search_value.lower()}%"
                }).fetchall()
            
            # Convert to options format
            options = []
            for gene_id, gene_name, gene_index in result:
                options.append({
                    'label': f"{gene_name} ({gene_id})",
                    'value': gene_index
                })
            
            return options
    except Exception as e:
        print(f"Error in database search: {e}")
        return []

def get_gene_data_with_metadata(gene_index, with_polars=True, limit=100):
    """
    Get complete data for a specific gene, joined with metadata using SQLAlchemy.
    
    Args:
        gene_index (str): The gene_index to query
        table_name (str): The matrix table name to query
        with_polars (bool): Whether to return a Polars DataFrame (True) or Pandas DataFrame (False)
        limit (int): Maximum number of rows to return (default 100)
    
    Returns:
        DataFrame: Either a Polars or Pandas DataFrame containing the joined data
        
    Raises:
        Exception: If there's an error in the query or if the gene is not found
    """
    
    # Create a cache key for this specific query
    cache_key = f"{gene_index}_all_transcript_data_{limit}_{with_polars}"
    
    # Check if we have this cached already
    if cache_key in MATRIX_DATA_CACHE:
        cached_data = MATRIX_DATA_CACHE[cache_key]
        return cached_data
    
    with pg_engine.connect() as conn:
        # Check if table exists
        table_exists_query = text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = 'all_transcript_data'
            )
        """)
        table_exists = conn.execute(table_exists_query, {"table_name": "all_transcript_data"}).scalar()
        
        if not table_exists:
            raise Exception(f"Table 'all_transcript_data' does not exist in the database")
        
        # Check if gene exists (use cache if possible)
        if gene_index in GENE_INFO_CACHE:
            gene_result = GENE_INFO_CACHE[gene_index]
        else:
            gene_query = text("""
                SELECT gene_index
                FROM gene_and_transcript_index_table 
                WHERE gene_index = :gene_index
                LIMIT 1
            """)
            gene_result = conn.execute(gene_query, {"gene_index": gene_index}).fetchone()
            
            if gene_result:
                GENE_INFO_CACHE[gene_index] = gene_result
        
        if not gene_result:
            raise Exception(f"Gene ID '{gene_index}' not found in annotation table")           
        
        ## Make the query
        query = f"""
        SELECT *
        FROM all_transcript_data AS at
        JOIN metadata      AS m
        USING (sample_id)
        WHERE gene_index = :gene_index
        """ 

        # Read directly to pandas
        df = pd.read_sql(text(query), conn, params={"gene_index": gene_index, "limit": limit})

        # Convert to polars if requested
        if with_polars and POLARS_AVAILABLE:
            df = pl.from_pandas(df)
        
        # Cache the result
        MATRIX_DATA_CACHE[cache_key] = df
        
        return df

def get_gene_density_data(gene_index):
    """
    Get gene expression data from the density plot table.
    
    Args:
        gene_index (str): The gene_index to query
        
    Returns:
        tuple: (log10(mean_cpm_normalized_tmm), expression_percentile) or (None, None) if gene not found
    """
    try:
        with pg_engine.connect() as conn:
            query = text("""
                SELECT "log10(mean_cpm_normalized_tmm)", expression_percentile 
                FROM gene_level_data_for_density_plot
                WHERE gene_index = :gene_index
                LIMIT 1
            """)
            result = conn.execute(query, {"gene_index": gene_index}).fetchone()
            
            if result:
                return result[0], result[1]
            return None, None
            
    except Exception as e:
        print(f"Error getting gene density data: {e}")
        return None, None

def get_total_gene_data_with_metadata(gene_index, with_polars=True, limit=None):
    """
    Get complete data for a specific gene from total_gene_data table, joined with metadata.
    
    Args:
        gene_index (str): The gene_index to query
        with_polars (bool): Whether to return a Polars DataFrame (True) or Pandas DataFrame (False)
        limit (int): Maximum number of rows to return (default None)
    
    Returns:
        DataFrame: Either a Polars or Pandas DataFrame containing the joined data
        
    Raises:
        Exception: If there's an error in the query or if the gene is not found
    """
    # Create a cache key for this specific query
    cache_key = f"total_{gene_index}_{limit}_{with_polars}"
    
    # Check if we have this cached already
    if cache_key in MATRIX_DATA_CACHE:
        cached_data = MATRIX_DATA_CACHE[cache_key]
        return cached_data


    with pg_engine.connect() as conn:
        # Check if gene exists (use cache if possible)
        if gene_index in GENE_INFO_CACHE:
            gene_result = GENE_INFO_CACHE[gene_index]
        else:
            gene_query = text("""
                SELECT gene_index
                FROM gene_and_transcript_index_table 
                WHERE gene_index = :gene_index
                LIMIT 1
            """)
            gene_result = conn.execute(gene_query, {"gene_index": gene_index}).fetchone()
            
            if gene_result:
                GENE_INFO_CACHE[gene_index] = gene_result
        
        if not gene_result:
            raise Exception(f"Gene ID '{gene_index}' not found in annotation table")
        
        
        ## Make the query
        query = f"""
        SELECT *
        FROM total_gene_data AS tg
        JOIN metadata      AS m
        USING (sample_id)
        WHERE gene_index = :gene_index
        """ 

        # Read directly to pandas
        df = pd.read_sql(text(query), conn, params={"gene_index": gene_index, "limit": limit})
     
        # Convert to polars if requested
        if with_polars and POLARS_AVAILABLE:
            df = pl.from_pandas(df)
        
        # Cache the result
        MATRIX_DATA_CACHE[cache_key] = df
        return df

def search_genes_tab1(search_value, previous_search=None):
    """
    Search for genes using the in-memory index.
    Modified version for tab1 that returns gene_name as the value.
    
    Args:
        search_value (str): The current search value
        previous_search (str): The previous search value, not used in this implementation
    """
    if not search_value:
        return []
    
    # Make sure gene index is loaded
    if not GENE_INDEX_LOADED:
        _load_gene_index()
    
    # If still no genes, fall back to database search
    if not ALL_GENES:
        return _search_genes_tab1_database(search_value)
    
    # Convert to lowercase once
    search_value = search_value.lower()
    
    # In-memory search
    filtered_results = []
    exact_matches = []
    prefix_matches = []
    contains_matches = []
    
    # First pass: categorize matches
    for gene_id, gene_name, gene_index in ALL_GENES:
        lower_id = gene_id.lower()
        lower_name = gene_name.lower()
        
        # Exact match (highest priority)
        if lower_id == search_value or lower_name == search_value:
            exact_matches.append((gene_id, gene_name, gene_index))
        # Prefix match (medium priority)
        elif lower_id.startswith(search_value) or lower_name.startswith(search_value):
            prefix_matches.append((gene_id, gene_name, gene_index))
        # Contains match (lowest priority)
        elif search_value in lower_id or search_value in lower_name:
            contains_matches.append((gene_id, gene_name, gene_index))
            
        # Stop when we have enough matches
        if len(exact_matches) >= 10:
            break
    
    # Combine results in priority order
    filtered_results = exact_matches
    
    # Add prefix matches if needed
    if len(filtered_results) < 10:
        filtered_results.extend(prefix_matches[:10 - len(filtered_results)])
    
    # Add contains matches if needed
    if len(filtered_results) < 10:
        filtered_results.extend(contains_matches[:10 - len(filtered_results)])
    
    # Convert to options format
    options = []
    for gene_id, gene_name, gene_index in filtered_results[:10]:
        options.append({
            'label': f"{gene_name} ({gene_id})",
            'value': gene_name  # Return gene_name instead of gene_index
        })
    
    return options

def _search_genes_tab1_database(search_value):
    """Fallback to database search if in-memory index fails - returns gene_name as value"""
    try:
        with pg_engine.connect() as conn:
            # For very short searches, just do a prefix match
            if len(search_value) < 3:
                query = text("""
                    SELECT DISTINCT gene_id, gene_name, gene_index
                    FROM gene_and_transcript_index_table 
                    WHERE LOWER(gene_id) LIKE :prefix
                    OR LOWER(gene_name) LIKE :prefix
                    LIMIT 10
                """)
                result = conn.execute(query, {"prefix": f"{search_value.lower()}%"}).fetchall()
            else:
                # For longer searches, do a more comprehensive search
                query = text("""
                    WITH ranked_results AS (
                        SELECT DISTINCT 
                            gene_id,
                            gene_name,
                            gene_index,
                            CASE 
                                WHEN LOWER(gene_id) = :exact THEN 1
                                WHEN LOWER(gene_name) = :exact THEN 2
                                WHEN LOWER(gene_id) LIKE :prefix THEN 3
                                WHEN LOWER(gene_name) LIKE :prefix THEN 4
                                ELSE 5
                            END as match_rank
                        FROM gene_and_transcript_index_table 
                        WHERE (
                            LOWER(gene_id) LIKE :pattern
                            OR LOWER(gene_name) LIKE :pattern
                        )
                    )
                    SELECT gene_id, gene_name, gene_index
                    FROM ranked_results
                    ORDER BY match_rank, gene_name
                    LIMIT 10
                """)
                
                result = conn.execute(query, {
                    "exact": search_value.lower(),
                    "prefix": f"{search_value.lower()}%",
                    "pattern": f"%{search_value.lower()}%"
                }).fetchall()
            
            # Convert to options format
            options = []
            for gene_id, gene_name, gene_index in result:
                options.append({
                    'label': f"{gene_name} ({gene_id})",
                    'value': gene_name  # Return gene_name instead of gene_index
                })
            
            return options
    except Exception as e:
        print(f"Error in database search: {e}")
        return []