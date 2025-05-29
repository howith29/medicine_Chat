import requests
import pandas as pd
import xml.etree.ElementTree as ET

api_key = ""

url = ""
params = {
    "servicekey" : api_key,
    "pageNo" : 1,
    "numOfRows" : 3,
    "type" : "xml"
}

res = requests.get(url, params=params)

root = ET.fromstring(res.content)
items = root.find('body').find('items')

data = []
for item in items:
    data.append({
        "제품명": item.findtext('itemName'),
        "업체명": item.findtext('entpName'),
        "효능": item.findtext('efcyQesitm'),
        "사용법": item.findtext('useMethodQesitm'),
        "부작용": item.findtext('seQesitm'),
        "주의사항": item.findtext('atpnQesitm'),
    })

df = pd.DataFrame(data)
print(df.head())

