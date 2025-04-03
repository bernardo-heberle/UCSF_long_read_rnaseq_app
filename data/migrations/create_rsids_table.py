from sqlalchemy import create_engine, text, inspect
import time

# Database connection configuration
db_url = "postgresql+psycopg2://postgres:isoforms@localhost:5432/ad_dash_app"
engine = create_engine(db_url, pool_size=5, max_overflow=10)

def create_rsids_table():
    """Create a new table containing unique RSIDs from the genotyping table."""
    try:
        print("Starting RSIDs table creation...")
        start_time = time.time()
        
        with engine.connect() as conn:
            # Start a transaction
            with conn.begin():
                # Drop the table if it exists
                print("Dropping existing rsids table if it exists...")
                drop_table_query = text("DROP TABLE IF EXISTS rsids;")
                conn.execute(drop_table_query)
                
                # Create the rsids table
                print("Creating rsids table...")
                create_table_query = text("""
                    CREATE TABLE rsids (
                        rsid VARCHAR(50) PRIMARY KEY
                    );
                """)
                conn.execute(create_table_query)
                
                # Insert unique RSIDs from genotyping table
                print("Inserting unique RSIDs from genotyping table...")
                insert_query = text("""
                    INSERT INTO rsids (rsid)
                    SELECT DISTINCT rsid
                    FROM genotyping
                    WHERE rsid IS NOT NULL
                    ORDER BY rsid;
                """)
                conn.execute(insert_query)
                
                # Create index on rsid column
                print("Creating index on rsid column...")
                create_index_query = text("""
                    CREATE INDEX idx_rsids_rsid 
                    ON rsids (rsid);
                """)
                conn.execute(create_index_query)
                
                # Get count of inserted RSIDs
                count_query = text("SELECT COUNT(*) FROM rsids;")
                result = conn.execute(count_query).scalar()
                
                end_time = time.time()
                duration = end_time - start_time
                
                print(f"Successfully created rsids table with {result:,} unique RSIDs")
                print(f"Operation completed in {duration:.2f} seconds")
        
        # Verify the table was actually created
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        if 'rsids' in tables:
            print("Verification: rsids table exists in the database")
            with engine.connect() as conn:
                verify_count = conn.execute(text("SELECT COUNT(*) FROM rsids;")).scalar()
                print(f"Verification: rsids table contains {verify_count:,} rows")
                if verify_count == 0:
                    print("WARNING: rsids table exists but contains no data!")
        else:
            raise Exception("Verification failed: rsids table was not created!")
                
    except Exception as e:
        print(f"Error creating rsids table: {e}")
        raise

if __name__ == "__main__":
    create_rsids_table() 