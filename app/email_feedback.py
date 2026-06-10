import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import smtplib
import os
import datetime
from dotenv import load_dotenv
# Load environment variables
load_dotenv(".env")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s (%(filename)s:%(lineno)s)',
    handlers=[
        logging.FileHandler("app.log", mode='a'),
        logging.StreamHandler()
    ]
)

class EmailFeedback:
    # Class-level configuration (shared across all instances)
    SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
    SMTP_PORT = int(os.getenv('SMTP_PORT', 587))
    SENDER_EMAIL = os.getenv('SENDER_EMAIL')
    SENDER_PASSWORD = os.getenv('SENDER_PASSWORD')
    RECIPIENT_EMAIL = os.getenv('RECIPIENT_EMAIL', SENDER_EMAIL)
    
    @classmethod
    def compose_email(cls, notification_type: str, message: str, image_path: str = None):
        # Validate credentials before proceeding
        if not cls.SENDER_EMAIL or not cls.SENDER_PASSWORD:
            logging.error("Email credentials not configured. Check .env file.")
            return False

        try:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            color = "#4CAF50" if notification_type.lower() == "success" else "#f44336"
            
            # Create email container
            msg = MIMEMultipart()
            msg['Subject'] = f'Facial Recognition: {notification_type} - {timestamp}'
            msg['From'] = cls.SENDER_EMAIL
            msg['To'] = cls.RECIPIENT_EMAIL

            # Create HTML email body
            body = f"""<html>
<body style="font-family: Arial, sans-serif;">
    <div style="border-left: 4px solid {color}; padding-left: 15px;">
        <h2 style="color: {color}; margin-bottom: 5px;">{notification_type} Notification</h2>
        <p><strong>System:</strong> Facial Recognition</p>
        <p><strong>Timestamp:</strong> {timestamp}</p>
        <div style="background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin: 10px 0;">
            <p style="margin-top: 0;"><strong>Details:</strong></p>
            <pre style="white-space: pre-wrap; font-family: inherit;">{message}</pre>
        </div>
        <p style="font-size: 0.9em; color: #666;">
            This is an automated notification. Please check application logs for full context.
        </p>
    </div>
</body>
</html>"""
            
            msg.attach(MIMEText(body, "html"))
            
            # Attach image if provided
            if image_path:
                try:
                    if os.path.exists(image_path):
                        with open(image_path, 'rb') as img_file:
                            img = MIMEImage(img_file.read())
                            img.add_header('Content-Disposition', 'attachment', 
                                          filename=os.path.basename(image_path))
                            msg.attach(img)
                    else:
                        logging.warning(f"Image not found: {image_path}")
                except Exception as img_error:
                    logging.error(f"Image attachment failed: {img_error}")

            # Send email
            with smtplib.SMTP(cls.SMTP_SERVER, cls.SMTP_PORT) as server:
                server.starttls()
                server.login(cls.SENDER_EMAIL, cls.SENDER_PASSWORD)
                server.send_message(msg)
                
            logging.info(f"Sent {notification_type} notification to {cls.RECIPIENT_EMAIL}")
            return True
        
        except Exception as e:
            logging.error(f"Email sending failed: {e}", exc_info=True)
            return False