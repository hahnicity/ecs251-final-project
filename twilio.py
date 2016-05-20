import ConfigParser

from twilio.access_token import AccessToken, IpMessagingGrant
from twilio.rest import TwilioRestClient

# You will need your Account Sid and a API Key Sid and Secret
# to generate an Access Token for your SDK endpoint to connect to Twilio.

IDENTITY = "noreply@ecs251.com"
DEVICE_ID = "sparks-in-a-can"

with open("/home/ec2-user/twilio.config") as f:
    parser = ConfigParser.ConfigParser()
    parser.readfp(f)
    account_sid = parser.get("config", "account_sid")
    apikey_sid = parser.get("config", "apikey_sid")
    apikey_secret = parser.get("config", "apikey_secret")
    service_sid = parser.get("config", "service_sid")


def get_token():
    token = AccessToken(apikey_sid, account_sid, apikey_secret, IDENTITY)
    endpoint_id = "HipFlowSlackDockRC:{0}:{1}".format(IDENTITY, DEVICE_ID)
    ipm_grant = IpMessagingGrant(endpoint_id=endpoint_id, service_sid=service_sid)
    token.add_grant(ipm_grant)
    return token.to_jwt()


def send_text(to, from_, token, body):
    client = TwilioRestClient(account_sid, token)
    client.messages.create(
        to=to,
        from_=from_,
        body=body
    )
