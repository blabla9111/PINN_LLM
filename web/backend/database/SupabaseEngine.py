from supabase import create_client, Client

class SupabaseEngine:
    def __init__(self, config):
        url = config.supabase.URL
        key = config.supabase.KEY
        if not url or not key:
            raise ValueError("Supabase credentials not found in config")
        
        self.supabase: Client = create_client(url, key)