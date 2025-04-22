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

# Database connection configuration
PG_HOST = 'localhost'
PG_PORT = '5432'
PG_DB = 'ad_dash_app'
PG_USER = 'postgres'
PG_PASSWORD = 'isoforms'

# Initialize SQLAlchemy engine - global connection
pg_engine = create_engine(f'postgresql://{PG_USER}:{PG_PASSWORD}@{PG_HOST}:{PG_PORT}/{PG_DB}')

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

# For RSID searching
ALL_RSIDS = {}  # Change to dictionary for faster lookups
RSID_INDEX_LOADED = False
RSID_INDEX_LOADING = False
RSID_SEARCH_CACHE = {}  # Add cache for search results

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.value = None

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, key, value):
        node = self.root
        for char in key:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
        node.value = value
    
    def find_prefix_matches(self, prefix, max_results=10):
        results = []
        node = self.root
        
        # Navigate to the prefix node
        for char in prefix:
            if char not in node.children:
                return results
            node = node.children[char]
        
        # Collect all matches from this node
        def collect_matches(node):
            if len(results) >= max_results:
                return
            if node.is_end:
                results.append(node.value)
            for child in node.children.values():
                collect_matches(child)
        
        collect_matches(node)
        return results

# Add after the global variables
RSID_TRIE = Trie()  # Trie for RSID prefix matching
SORTED_RSIDS = []  # Sorted list for binary search
RSID_BINARY_SEARCH_CACHE = {}  # Cache for binary search results

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

def get_matrix_tables():
    """Get all transcript data tables from the database using SQLAlchemy."""
    try:
        with pg_engine.connect() as conn:
            query = text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                AND LOWER(table_name) LIKE '%transcript_data%'
                ORDER BY table_name
            """)
            result = conn.execute(query).fetchall()
        
        tables = [row[0] for row in result]
        return tables
    except Exception as e:
        print(f"Error getting tables: {e}")
        return []

def get_matrix_dropdown_options():
    """Get formatted dropdown options for different counting methods."""
    # Define options for different counting methods that will use columns from all_transcript_data
    ordered_options = [
        {'label': 'Total Counts', 'value': 'total'},
        {'label': 'Unique Counts', 'value': 'unique'},
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

def get_table_info(table_name):
    """Get basic information about a table including dimensions and preview using SQLAlchemy."""
    try:
        with pg_engine.connect() as conn:
            # Get column count
            col_query = text(f"""
                SELECT COUNT(*) 
                FROM information_schema.columns 
                WHERE table_name = :table_name
            """)
            col_count = conn.execute(col_query, {"table_name": table_name}).scalar()
            
            # Get row count
            row_query = text(f"SELECT COUNT(*) FROM {table_name}")
            row_count = conn.execute(row_query).scalar()
            
            # Get first few rows preview
            preview_query = text(f"SELECT * FROM {table_name} LIMIT 5")
            preview_df = pd.read_sql(preview_query, conn)
            
            # Convert to Polars if available
            if POLARS_AVAILABLE:
                preview_df = pl.from_pandas(preview_df)
        
        return {
            'row_count': row_count,
            'col_count': col_count,
            'preview_df': preview_df
        }
    except Exception as e:
        raise Exception(f"Error getting table info: {str(e)}")


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

def clear_cache():
    """Clear all cached data"""
    global GENE_INFO_CACHE, MATRIX_DATA_CACHE, SEARCH_RESULTS_CACHE, ALL_GENES, GENE_INDEX_LOADED, GENE_INDEX_LOADING
    global ALL_RSIDS, RSID_INDEX_LOADED, RSID_INDEX_LOADING, RSID_SEARCH_CACHE, RSID_TRIE, SORTED_RSIDS, RSID_BINARY_SEARCH_CACHE
    
    GENE_INFO_CACHE = {}
    MATRIX_DATA_CACHE = {}
    SEARCH_RESULTS_CACHE = {}
    ALL_GENES = []
    GENE_INDEX_LOADED = False
    ALL_RSIDS = {}
    RSID_INDEX_LOADED = False
    RSID_SEARCH_CACHE = {}
    RSID_TRIE = Trie()
    SORTED_RSIDS = []
    RSID_BINARY_SEARCH_CACHE = {}
    # Don't reset GENE_INDEX_LOADING or RSID_INDEX_LOADING here as it might interfere with ongoing operations

# Cleanup function to call when the app shuts down
def cleanup():
    """Clean up resources when the app shuts down."""
    global pg_engine
    clear_cache()
    if pg_engine:
        pg_engine.dispose()

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


def binary_search_prefix(prefix, max_results=10):
    """Binary search for prefix matches in sorted RSIDs list."""
    if not SORTED_RSIDS:
        return []
    
    # Check cache first
    if prefix in RSID_BINARY_SEARCH_CACHE:
        return RSID_BINARY_SEARCH_CACHE[prefix]
    
    results = []
    left, right = 0, len(SORTED_RSIDS) - 1
    
    # Find the first occurrence of the prefix
    while left <= right:
        mid = (left + right) // 2
        if SORTED_RSIDS[mid].startswith(prefix):
            # Found a match, now collect all matches
            results.append(ALL_RSIDS[SORTED_RSIDS[mid]])
            # Check left side
            left_idx = mid - 1
            while left_idx >= 0 and SORTED_RSIDS[left_idx].startswith(prefix):
                results.append(ALL_RSIDS[SORTED_RSIDS[left_idx]])
                left_idx -= 1
            # Check right side
            right_idx = mid + 1
            while right_idx < len(SORTED_RSIDS) and SORTED_RSIDS[right_idx].startswith(prefix):
                results.append(ALL_RSIDS[SORTED_RSIDS[right_idx]])
                right_idx += 1
            break
        elif SORTED_RSIDS[mid] < prefix:
            left = mid + 1
        else:
            right = mid - 1
    
    # Limit results
    results = results[:max_results]
    
    # Cache the results
    RSID_BINARY_SEARCH_CACHE[prefix] = results
    return results

def search_rsids(search_value, previous_search=None):
    """
    Search for RSIDs using optimized in-memory index.
    Only performs prefix matching for RSIDs.
    
    Args:
        search_value (str): The current search value
        previous_search (str): The previous search value, not used in this implementation
    """
    if not search_value:
        return []
    
    # Check cache first
    cache_key = search_value.lower()
    if cache_key in RSID_SEARCH_CACHE:
        return RSID_SEARCH_CACHE[cache_key]
    
    # If no RSIDs, fall back to database search
    if not ALL_RSIDS:
        return _search_rsids_database(search_value)
    
    # Convert to lowercase once
    search_value = search_value.lower()
    
    # Use dictionary lookups for exact matches (O(1))
    exact_matches = []
    if search_value in ALL_RSIDS:
        exact_matches.append(ALL_RSIDS[search_value])
    
    # Use trie for prefix matches (very fast for prefix matching)
    prefix_matches = RSID_TRIE.find_prefix_matches(search_value, max_results=10 - len(exact_matches))
    
    # If trie doesn't find enough matches, try binary search as fallback
    if len(prefix_matches) < 10 - len(exact_matches):
        additional_matches = binary_search_prefix(search_value, max_results=10 - len(exact_matches) - len(prefix_matches))
        prefix_matches.extend(additional_matches)
    
    # Combine results in priority order
    filtered_results = exact_matches + prefix_matches
    
    # Convert to options format
    options = []
    for rsid in filtered_results[:10]:
        options.append({
            'label': rsid,
            'value': rsid
        })
    
    # Cache the results
    RSID_SEARCH_CACHE[cache_key] = options
    return options

def _search_rsids_database(search_value):
    """Fallback to database search if in-memory index fails"""
    try:
        with pg_engine.connect() as conn:
            # Optimized query using index hints
            query = text("""
                WITH rsid_matches AS (
                    SELECT rsid, rsid_index
                    FROM rsid_index_table
                    WHERE rsid IS NOT NULL
                    AND LOWER(rsid) LIKE :prefix || '%'
                )
                SELECT rsid, rsid_index
                FROM rsid_matches
                ORDER BY rsid
                LIMIT 10
            """)
            
            # Execute query with parameter binding
            result = conn.execute(query, {"prefix": search_value.lower()})
            return [{"label": row[0], "value": row[1]} for row in result]
    except Exception as e:
        print(f"Error in RSID database search: {e}")
        return []

def get_rsid_data(rsid_index, with_polars=True):
    """
    Get data for a specific RSID from the genotyping table, joined with metadata tables.

    Args:
        rsid_index (str): The rsid_index to query
        with_polars (bool): Whether to return a Polars DataFrame (True) or Pandas DataFrame (False)

    Returns:
        DataFrame: Either a Polars or Pandas DataFrame containing the joined genotyping data

    Raises:
        Exception: If there's an error in the query or if the RSID index is not found
    """
    # Return empty DataFrame if rsid_index is None
    if rsid_index is None:
        if with_polars and POLARS_AVAILABLE:
            return pl.DataFrame()
        return pd.DataFrame()

    # Create a cache key for this specific query
    cache_key = f"rsid_{rsid_index}_{with_polars}"

    # Check if we have this cached already
    if cache_key in MATRIX_DATA_CACHE:
        cached_data = MATRIX_DATA_CACHE[cache_key]
        return cached_data

    try:
        with pg_engine.connect() as conn:
            # Build the main query with joins - filter first then join
            query = text("""
                WITH filtered_genotyping AS (
                    SELECT sample_id, genotype_index, rsid_index
                    FROM genotyping
                    WHERE rsid_index = :rsid_index
                )
                SELECT
                    g.sample_id,
                    ri.rsid,
                    gi.genotype
                    -- Add other columns from genotyping table if needed, e.g., g.some_other_column
                FROM filtered_genotyping g
                JOIN rsid_index_table ri ON g.rsid_index = ri.rsid_index
                JOIN genotype_index_table gi ON g.genotype_index = gi.genotype_index
            """)

            # Read directly to pandas
            df = pd.read_sql(query, conn, params={"rsid_index": rsid_index})

            if df.empty:
                # Check if the rsid_index actually exists in the index table
                check_query = text("SELECT 1 FROM rsid_index_table WHERE rsid_index = :rsid_index LIMIT 1")
                exists = conn.execute(check_query, {"rsid_index": rsid_index}).scalar()
                if exists:
                    # If the index exists but no data, return empty DataFrame (maybe log a warning)
                    print(f"Warning: rsid_index '{rsid_index}' found in index table but has no data in genotyping table.")
                    if with_polars and POLARS_AVAILABLE:
                        df = pl.DataFrame({'sample_id': [], 'rsid': [], 'genotype': []})
                    else:
                        df = pd.DataFrame({'sample_id': [], 'rsid': [], 'genotype': []})

                else:
                     raise Exception(f"rsid_index '{rsid_index}' not found in rsid_index_table")


            # Convert to polars if requested
            if with_polars and POLARS_AVAILABLE:
                df = pl.from_pandas(df)

            # Cache the result
            MATRIX_DATA_CACHE[cache_key] = df
            return df

    except Exception as e:
        print(f"Error getting RSID data for rsid_index {rsid_index}: {e}")
        # Fallback is less useful here as the primary issue might be joins or missing index
        # Reraise the exception for better debugging
        raise Exception(f"Error processing rsid_index '{rsid_index}': {str(e)}")