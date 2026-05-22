import os
import sys
import traceback
from dotenv import load_dotenv

load_dotenv()
from supabase import create_client

url = os.environ.get('SUPABASE_URL')
key = os.environ.get('SUPABASE_KEY')

print(f"URL: {url}")
print(f"KEY: {key[:10]}...")

try:
    supabase = create_client(url, key)
    print("Success:", supabase)
except Exception as e:
    print("FAILED")
    with open("err.txt", "w") as f:
        traceback.print_exc(file=f)
