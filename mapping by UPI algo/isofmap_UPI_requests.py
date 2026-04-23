#!/usr/bin/env python3
"""
ISOMAP POST Request Script
Прави POST заявка към http://www.isofmap.bg/ за търсене на парцели
и след това WFS заявка за получаване на координатите
"""

import requests
import json
import re
import time
import random
from urllib.parse import parse_qs, urlparse
from bs4 import BeautifulSoup

def extract_adm_id_from_html(html_content):
    """
    Извлича adm_id от HTML отговора за УПИ търсене
    """
    try:
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Намери таблицата за searchRegParcel (УПИ)
        parcel_table = soup.find('table', {'id': 'searchResultTable9'})
        if not parcel_table:
            return None
            
        # Намери първия ред с данни
        tbody = parcel_table.find('tbody')
        if not tbody:
            return None
            
        first_row = tbody.find('tr')
        if not first_row:
            return None
            
        # Първата колона съдържа adm_id
        first_cell = first_row.find('td', {'class': 'select-feature'})
        if not first_cell:
            return None
            
        adm_id = first_cell.get_text().strip()
        return adm_id
        
    except Exception as e:
        return None

def make_wfs_request(adm_id, cookies, upi_number):
    """
    Прави WFS заявка за получаване на GeoJSON координати
    
    Args:
        adm_id (str): Административен ID на парцела 
        cookies (dict): Cookies от първата заявка
        upi_number (str): УПИ номер за показване в резултата
        
    Returns:
        dict или None: GeoJSON данни ако успешно, иначе None
    """
    
    url = "http://www.isofmap.bg/owsmap"
    
    # WFS XML заявка
    wfs_xml = f'''<GetFeature xmlns="http://www.opengis.net/wfs" service="WFS" version="1.1.0" outputFormat="geojson" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.opengis.net/wfs http://schemas.opengis.net/wfs/1.1.0/wfs.xsd"><Query typeName="ms:reg_parcel" srsName="EPSG:7801" xmlns:ms="http://mapserver.gis.umn.edu/mapserver"><Filter xmlns="http://www.opengis.net/ogc"><PropertyIsEqualTo><PropertyName>adm_id</PropertyName><Literal>{adm_id}</Literal></PropertyIsEqualTo></Filter></Query></GetFeature>'''
    
    headers = {
        'Accept': '*/*',
        'Accept-Encoding': 'gzip, deflate',
        'Accept-Language': 'bg-BG,bg;q=0.9,en-US;q=0.8,en;q=0.7,ar;q=0.6',
        'Connection': 'keep-alive',
        'Content-Type': 'application/xml; charset=UTF-8',
        'Host': 'www.isofmap.bg',
        'Origin': 'http://www.isofmap.bg',
        'Referer': 'http://www.isofmap.bg/',
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36',
        'X-Requested-With': 'XMLHttpRequest'
    }
    
    try:
        response = requests.post(
            url=url,
            data=wfs_xml,
            headers=headers,
            cookies=cookies,
            timeout=30
        )
        
        if response.status_code == 200:
            try:
                geojson_data = response.json()
                
                # Извлечи координатите ако има features
                if 'features' in geojson_data and len(geojson_data['features']) > 0:
                    feature = geojson_data['features'][0]
                    if 'geometry' in feature and 'coordinates' in feature['geometry']:
                        coordinates = feature['geometry']['coordinates']
                        # Форматирай като поискано: <УПИ> : координати
                        print(f"<{upi_number}> : {coordinates}")
                        
                return geojson_data
                
            except json.JSONDecodeError:
                print("Не е JSON отговор от WFS заявката")
                return None
        else:
            print(f"WFS грешка: HTTP {response.status_code}")
            return None
            
    except Exception as e:
        print(f"Грешка при WFS заявката: {e}")
        return None

def make_isomap_post_request(upi_number="XVII 1209, 1"):
    """
    Прави POST заявка към ISOMAP системата и след това WFS заявка за координати
    
    Args:
        upi_number (str): УПИ номер за търсене (напр. "XVII 1209, 1")
    """
    
    # URL за заявката
    url = "http://www.isofmap.bg/search"
    
    # Параметри като променливи
    search_type = "searchFast"
    search_fast = upi_number
    token = "77f989a4b5d1487c78fa432cb58bdb9.3tP_AsTZS4VT-CSrpqAAH9iZGW1x-HcjHS0l2yfC3OQ.5qqLe5S6IPUZi2KGzfQxV4yhVBQ_rTAWLVdyvkylqruamo1otJsItgyZHA"
    
    # POST данни (в точния ред от браузъра)
    post_data = {
        'token': token,
        'searchType': search_type,
        'searchFast': search_fast
    }
    
    # Headers които точно симулират браузър заявката
    headers = {
        'Accept': 'text/html, */*; q=0.01',
        'Accept-Encoding': 'gzip, deflate',
        'Accept-Language': 'bg-BG,bg;q=0.9,en-US;q=0.8,en;q=0.7,ar;q=0.6',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
        'Host': 'www.isofmap.bg',
        'Origin': 'http://www.isofmap.bg',
        'Pragma': 'no-cache',
        'Referer': 'http://www.isofmap.bg/',
        'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36',
        'X-Requested-With': 'XMLHttpRequest'
    }
    
    # Cookies (важни за сесията)
    cookies = {
        'PHPSESSID': 'f58rjrnpmnp5d5ddejsbfgnej5',
        '_ga': 'GA1.1.979076318.1756761046',
        '_ga_JYN8Q1VR0D': 'GS2.1.1756761046.1.1.1756762401.60.0.0'
    }

    try:
        print(f"Търсене на УПИ: {upi_number}")
        
        # Правене на POST заявката
        response = requests.post(
            url=url,
            data=post_data,
            headers=headers,
            cookies=cookies,
            timeout=30,
            allow_redirects=True
        )
        
        # Показване само на статуса
        print(f"Статус: {response.status_code}")
        
        # Показване на отговора
        if response.status_code == 200:
            try:
                json_response = response.json()
                if json_response.get('success'):
                    print(f"Намерени резултати: {json_response.get('count', 0)}")
                else:
                    print(f"Грешка: {json_response.get('error', 'Unknown error')}")
                    
            except json.JSONDecodeError:
                # Извличане на adm_id от HTML
                adm_id = extract_adm_id_from_html(response.text)
                
                if adm_id:
                    # Правене на WFS заявка за координати
                    geojson_data = make_wfs_request(adm_id, cookies, upi_number)
                    
                    if geojson_data:
                        return {
                            'search_response': response.text,
                            'adm_id': adm_id,
                            'geojson': geojson_data,
                            'upi': upi_number
                        }
                    else:
                        print("Не успях да получа координати от WFS заявката")
                else:
                    print("Не успях да извлека adm_id от HTML отговора")
        else:
            print(f"Грешка: HTTP {response.status_code}")
            
        return response
        
    except requests.exceptions.Timeout:
        print("Грешка: Timeout - заявката отне твърде много време")
        return None
    except requests.exceptions.ConnectionError:
        print("Грешка: Проблем с връзката")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Грешка при заявката: {e}")
        return None

def process_upi_list(upi_list):
    """
    Обработва списък с УПИ номера със случайни паузи между заявките
    
    Args:
        upi_list (list): Списък с УПИ номера за търсене
    """
    
    print(f"Ще се обработят {len(upi_list)} УПИ номера:")
    for i, upi in enumerate(upi_list):
        print(f"  {i+1}. {upi}")
    print("-" * 50)
    
    results = []
    
    for i, upi_number in enumerate(upi_list):
        print(f"\n[{i+1}/{len(upi_list)}]")
        
        try:
            # Направи заявката за текущия УПИ
            result = make_isomap_post_request(upi_number)
            results.append({
                'upi': upi_number,
                'result': result
            })
        except Exception as e:
            print(f"Грешка при обработка на {upi_number}: {e}")
            results.append({
                'upi': upi_number,
                'result': None,
                'error': str(e)
            })
        
        # Добави пауза само ако не е последния елемент
        if i < len(upi_list) - 1:
            wait_time = random.uniform(1.0, 1.5)
            print(f"Пауза: {wait_time:.2f} сек...")
            time.sleep(wait_time)
    
    print(f"\nОбработени всички {len(upi_list)} УПИ номера.")
    return results

# ГЛАВНА ЧАСТ НА СКРИПТА
if __name__ == "__main__":
    print("ISOMAP УПИ Търсене")
    print("=" * 30)
    
    # Проверка за необходими пакети
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        print("BeautifulSoup не е инсталиран!")
        print("Инсталирайте го с: pip install beautifulsoup4")
        exit(1)
    
    # СПИСЪК С УПИ НОМЕРА - ПРОМЕНЕТЕ ТЕЗИ СТОЙНОСТИ:
    upi_list = [
        "VIII 1763",
        "XVIII 5678, 2", 
        "XVIII 5780 2",
        "XVIII 5558, 2"
    ]
    
    # Обработка на всички УПИ със случайни паузи
    results = process_upi_list(upi_list)
    
    print("\nГотово.")