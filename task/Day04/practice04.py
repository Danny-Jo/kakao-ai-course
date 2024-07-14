import re

email = "MY name is Jhin, my email is kakaotech@goorm.io"
id_list = ["jhin.lee", "lovelove123", "세종대왕만세!!", "twin에너지123", "PostModern"]
html = "<p>Hello, <b>World!</b></p>"

email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
email_match = re.search(email_pattern, email)
email = email_match.group()
print(email)

def anonymize_id(id):
    if len(id) > 3:
        return id[:3] + '*' * (len(id) - 3)
    return id
anonymized_ids = [anonymize_id(id) for id in id_list]
print("Anonymized IDs:", anonymized_ids)

html_pattern = re.compile(r'<.*?>')
html = re.sub(html_pattern, '', html)
print("Clean html:", html)