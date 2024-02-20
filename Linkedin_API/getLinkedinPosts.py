
from linkedin import linkedin

APPLICATON_KEY    = '77nzvvp4e0bhr0'
APPLICATON_SECRET = 'OeD2MoCHnkXZhCX2'

#RETURN_URL = 'http://localhost:8000'
RETURN_URL = 'https://www.linkedin.com/developers/tools/oauth/redirect'

authentication = linkedin.LinkedInAuthentication(
                    APPLICATON_KEY,
                    APPLICATON_SECRET,
                    RETURN_URL,
                    linkedin.PERMISSIONS.enums.values()

result = authentication.get_access_token()

print ("Access Token:", result.access_token)
print ("Expires in (seconds):", result.expires_in)
# Optionally one can send custom "state" value that will be returned from OAuth server
# It can be used to track your user state or something else (it's up to you)
# Be aware that this value is sent to OAuth server AS IS - make sure to encode or hash it
#authorization.state = 'your_encoded_message'

print (authentication.authorization_url)  # open this url on your browser
application = linkedin.LinkedInApplication(token=str(result.access_token))
application.search_company(selectors=[{'companies': ['name', 'universal-name', 'website-url']}], params={'keywords': 'apple microsoft'})
# Search URL is https://api.linkedin.com/v1/company-search:(companies:(name,universal-name,website-url))?keywords=apple%20microsoft

#from linkedin import server
#application = server.quick_api('77nj4j9sd6mspu', 'yQIsbcxuOnYPAY1I')
"""
r'''
from linkedin import linkedin

# Define CONSUMER_KEY, CONSUMER_SECRET,
# USER_TOKEN, and USER_SECRET from the credentials
# provided in your LinkedIn application

# Instantiate the developer authentication class

authentication = linkedin.LinkedInDeveloperAuthentication(
                   CONSUMER_KEY='77nzvvp4e0bhr0'
                    CONSUMER_SECRET='OeD2MoCHnkXZhCX2',
                    USER_TOKEN,
                    USER_SECRET,
                    RETURN_URL,
                    linkedin.PERMISSIONS.enums.values()
                )

# Optionally one can send custom "state" value that will be returned from OAuth server
# It can be used to track your user state or something else (it's up to you)
# Be aware that this value is sent to OAuth server AS IS - make sure to encode or hash it

# authorization.state = 'your_encoded_message'

# Pass it in to the app...

application = linkedin.LinkedInApplication(authentication)

# Use the app....

application.get_profile()
'''
"""