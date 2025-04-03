from sqlalchemy import create_engine, text, inspect

# Database connection configuration
db_url = "postgresql+psycopg2://postgres:isoforms@localhost:5432/ad_dash_app"
engine = create_engine(db_url, pool_size=5, max_overflow=10)

def check_rsids_table():
    """Check if the rsids table exists and show its structure."""
    try:
        print("Checking database structure...")
        
        # Create an inspector
        inspector = inspect(engine)
        
        # Get all table names
        tables = inspector.get_table_names()
        print("\nAll tables in database:")
        for table in tables:
            print(f"- {table}")
        
        # Check if rsids table exists
        if 'rsids' in tables:
            print("\nrsids table exists!")
            
            # Get columns in rsids table
            columns = inspector.get_columns('rsids')
            print("\nColumns in rsids table:")
            for column in columns:
                print(f"- {column['name']}: {column['type']}")
            
            # Get indexes on rsids table
            indexes = inspector.get_indexes('rsids')
            print("\nIndexes on rsids table:")
            for index in indexes:
                print(f"- {index['name']}: {index['column_names']}")
            
            # Get row count
            with engine.connect() as conn:
                count = conn.execute(text("SELECT COUNT(*) FROM rsids")).scalar()
                print(f"\nNumber of RSIDs in table: {count:,}")
        else:
            print("\nrsids table does not exist!")
            
    except Exception as e:
        print(f"Error checking database structure: {e}")
        raise

if __name__ == "__main__":
    check_rsids_table() 