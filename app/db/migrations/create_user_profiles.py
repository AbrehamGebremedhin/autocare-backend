"""
Migration script to create user_profiles table for storing additional user data
beyond what's in Supabase auth.users table.
"""
from app.db.base import SupabaseDBHandler
from app.utils.logger import get_logger_instance
import asyncio

logger = get_logger_instance("migration")

async def create_user_profiles_table():
    """Create user_profiles table if it doesn't exist"""
    try:
        db_handler = SupabaseDBHandler()
        
        async with db_handler.get_connection() as db:
            # Check if table already exists
            try:
                existing_check = db.table('user_profiles').select('id').limit(1).execute()
                await logger.info("user_profiles table already exists")
                return True
            except Exception:
                # Table doesn't exist, we'll create it
                pass
            
            # Create the user_profiles table using individual SQL commands
            # Note: Supabase doesn't allow direct DDL through the client, 
            # so this would need to be done through the Supabase dashboard or SQL editor
            
            await logger.info("user_profiles table needs to be created manually in Supabase dashboard")
            await logger.info("Please run the following SQL in your Supabase SQL editor:")
            
            sql_commands = """
-- Create user_profiles table
CREATE TABLE IF NOT EXISTS user_profiles (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE UNIQUE NOT NULL,
    cars TEXT[] DEFAULT '{}',
    preferences JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create index on user_id for faster lookups
CREATE INDEX IF NOT EXISTS idx_user_profiles_user_id ON user_profiles(user_id);

-- Enable RLS
ALTER TABLE user_profiles ENABLE ROW LEVEL SECURITY;

-- Drop existing policies if they exist (to avoid conflicts)
DROP POLICY IF EXISTS "Users can view own profile" ON user_profiles;
DROP POLICY IF EXISTS "Users can insert own profile" ON user_profiles;
DROP POLICY IF EXISTS "Users can update own profile" ON user_profiles;
DROP POLICY IF EXISTS "Admins can manage all profiles" ON user_profiles;

-- Create RLS policies
CREATE POLICY "Users can view own profile" ON user_profiles
    FOR SELECT USING (auth.uid() = user_id);
    
CREATE POLICY "Users can insert own profile" ON user_profiles
    FOR INSERT WITH CHECK (auth.uid() = user_id);
    
CREATE POLICY "Users can update own profile" ON user_profiles
    FOR UPDATE USING (auth.uid() = user_id);
    
-- Admin users can access all profiles  
CREATE POLICY "Admins can manage all profiles" ON user_profiles
    FOR ALL USING (
        EXISTS (
            SELECT 1 FROM auth.users 
            WHERE auth.users.id = auth.uid() 
            AND auth.users.raw_app_meta_data->>'role' = 'admin'
        )
    );
"""
            
            # Store SQL commands for manual execution if needed
            # This would normally be logged or returned for manual intervention
            
            return False  # Return False to indicate manual intervention needed
            
    except Exception as e:
        await logger.error(f"Error checking user_profiles table: {str(e)}")
        return False

async def main():
    """Run the migration"""
    await logger.info("Starting user_profiles table migration...")
    
    # Create the table
    table_created = await create_user_profiles_table()
    
    if table_created:
        await logger.info("Migration completed successfully")
    else:
        await logger.error("Migration failed")

if __name__ == "__main__":
    asyncio.run(main())
