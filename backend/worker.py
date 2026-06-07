import pika
import time
import os
import sys
from dotenv import load_dotenv

load_dotenv()

# Reads the URL from your local .env file (which is not pushed to GitHub)
AMQP_URL = os.getenv("AMQP_URL")

if not AMQP_URL:
    print("[-] ERROR: Missing AMQP_URL variable in .env file!")
    sys.exit(1)

def callback(ch, method, properties, body):
    print("\n[+] DING! Received a new task from Azure Cloud!")
    print("[+] Worker: Analyzing image via AI (this will take a moment)...")
    
    time.sleep(4) # Simulating heavy AI workload
    
    print("[+] Worker: Success! Glaucoma detected (CDR = 0.65)")
    print("[*] Worker: Waiting for more tasks...\n")
    
    # Manual acknowledgment after successful processing
    ch.basic_ack(delivery_tag=method.delivery_tag)

print("=====================================================")
print(" [*] DISTRIBUTED WORKER (Running locally on laptop)")
print(" [*] Connecting to CloudAMQP...")

try:
    params = pika.URLParameters(AMQP_URL)
    connection = pika.BlockingConnection(params)
    channel = connection.channel()

    channel.queue_declare(queue='glaucoma_queue')
    
    print(" [*] Connected! Listening on 'glaucoma_queue'...")
    print("=====================================================")

    # auto_ack=False ensures messages aren't lost if the worker crashes
    channel.basic_consume(queue='glaucoma_queue', on_message_callback=callback, auto_ack=False)
    channel.start_consuming()
    
except Exception as e:
    import traceback
    print(f"\n[-] Queue connection error occurred.")
    print(traceback.format_exc())
except KeyboardInterrupt:
    print("\n[*] Worker stopped by user.")
    if 'connection' in locals() and connection.is_open:
        connection.close()