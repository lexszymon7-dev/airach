import streamlit as st
import os
import shutil
import zipfile
import datetime
import json
import re
import time
import io
import unicodedata
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import dbf
from google import genai
from google.genai import types

st.set_page_config(page_title="KSP AI Agent Pro", page_icon="⚖️", layout="wide")

# Dodany napis na samej górze
st.markdown("### **AIrach - pamiętaj, że księgowy jest od picia kawy!**")
st.markdown("---")

KATALOG_FIRM = "baza_firm"
KATALOG_CACHE = "cache_faktur"
os.makedirs(KATALOG_FIRM, exist_ok=True)
os.makedirs(KATALOG_CACHE, exist_ok=True)

API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))

def pobierz_liste_firm():
    pliki = [f for f in os.listdir(KATALOG_FIRM) if f.endswith('.json')]
    firmy = {}
    for p in pliki:
        try:
            with open(os.path.join(KATALOG_FIRM, p), 'r', encoding='utf-8') as f:
                dane = json.load(f)
                firmy[dane['nip']] = dane
        except Exception:
            continue
    if not firmy:
        domyslna = {
            "nazwa": "Firma Testowa", 
            "nip": "5170297177", 
            "pkd_glowne": "62.01.Z", 
            "pkd_dodatkowe": "47.91.Z", 
            "instrukcje_specjalne": "Firma zajmuje się usługami IT oraz handlem elektronicznym.", 
            "reguly_kontrahentow": {}
        }
        zapisz_profil_firmy(domyslna)
        firmy[domyslna['nip']] = domyslna
    return firmy

def zapisz_profil_firmy(dane_firmy):
    with open(os.path.join(KATALOG_FIRM, f"{dane_firmy['nip']}.json"), 'w', encoding='utf-8') as f:
        json.dump(dane_firmy, f, ensure_ascii=False, indent=2)

baza_firm = pobierz_liste_firm()

with st.sidebar:
    st.header("⚡ Konfiguracja KSP i Agentów AI")
    if API_KEY: st.success("🟢 API Gemini: Aktywne")
    else: st.error("🔴 Brak klucza API Gemini!")
        
    opcje_firm = {f"{v['nazwa']} (NIP: {k})": k for k, v in baza_firm.items()}
    wybrana_etykieta = st.selectbox("Aktywna firma:", list(opcje_firm.keys()))
    aktywny_nip = opcje_firm[wybrana_etykieta]
    firma = baza_firm[aktywny_nip]

    with st.expander("⚙️ Profil i Agent AI tej firmy"):
        nowa_nazwa = st.text_input("Nazwa firmy", value=firma.get('nazwa', ''))
        nowy_pkd = st.text_input("PKD główne", value=firma.get('pkd_glowne', ''))
        nowe_instrukcje = st.text_area("Instrukcje dla Agent AI / Specyfika kosztów", value=firma.get('instrukcje_specjalne', ''))
        if st.button("Zapisz profil firmy"):
            firma['nazwa'] = nowa_nazwa
            firma['pkd_glowne'] = nowy_pkd
            firma['instrukcje_specjalne'] = nowe_instrukcje
            zapisz_profil_firmy(firma)
            st.success("Zapisano profil firmy!")
            st.rerun()

    with st.expander("➕ Dodaj nową firmę"):
        n_nazwa = st.text_input("Nazwa nowej firmy")
        n_nip = st.text_input("NIP nowej firmy")
        n_pkd = st.text_input("PKD główne (np. 47.91.Z)")
        if st.button("Utwórz profil firmy"):
            if n_nip and n_nazwa:
                nowa_f = {
                    "nazwa": n_nazwa, 
                    "nip": n_nip.replace("-", "").strip(), 
                    "pkd_glowne": n_pkd, 
                    "pkd_dodatkowe": "", 
                    "instrukcje_specjalne": "", 
                    "reguly_kontrahentow": {}
                }
                zapisz_profil_firmy(nowa_f)
                st.success(f"Utworzono firmę {n_nazwa}!")
                st.rerun()
            else:
                st.error("Podaj nazwę i NIP.")

    liczba_watkow = st.slider("Wątki równoległe OCR:", min_value=1, max_value=25, value=10)

    if st.button("🗑️ Wyczyść cache OCR"):
        if os.path.exists(KATALOG_CACHE):
            shutil.rmtree(KATALOG_CACHE)
            os.makedirs(KATALOG_CACHE)
            st.success("Wyczyszczono cache!")

def buduj_prompt_ekstrakcji(dane_firmy):
    reguly_str = json.dumps(dane_firmy.get('reguly_kontrahentow', {}), ensure_ascii=False)
    return f"""
Jesteś dedykowanym Agentem AI i opiekunem księgowym dla firmy "{dane_firmy['nazwa']}" (NIP: "{dane_firmy['nip']}").
Profil działalności firmy (PKD): {dane_firmy.get('pkd_glowne', 'Brak')}.
Dodatkowe instrukcje księgowe i kontekst firmy: {dane_firmy.get('instrukcje_specjalne', 'Brak')}.
Znane reguly przypisania dla kontrahentów: {reguly_str}.

Twoim zadaniem jest dokładna analiza faktury i poprawna kategoryzacja kosztów lub przychodów zgodnie z profilem tej konkretnej firmy.
Zwróć obiekt JSON z polami:
- kierunek ("ZAKUP" / "SPRZEDAZ")
- kategoria_pkpir ("KOLUMNA_10_TOWARY", "KOLUMNA_13_POZOSTALE", "SAMOCHOD_MIESZANY", "KOLUMNA_7_PRZYCHOD")
- typ_dokumentu ("Faktura VAT", "Dowód wewnętrzny", itp.)
- czy_zaplacono (true / false)
- nr_ksef (35 znaków lub null)
- data_wplywu_ksef (YYYY-MM-DD lub null)
- nr_dokumentu
- data_wystawienia (YYYY-MM-DD)
- data_sprzedazy (YYYY-MM-DD)
- termin_platnosci (YYYY-MM-DD - faktyczny ostateczny termin płatności)
- kontrahent (obiekt: nazwa, nip, ulica, kod, miasto, adres)
- kwoty (obiekt: netto (liczba), vat (liczba), brutto (liczba), stawka_vat (liczba lub tekst np. 23, 8, 5, 0, "zw"))
- opis_gospodarczy (do 30 znaków)
"""

def wykonaj_ekstrakcje_pdf(nazwa_pliku, bajty_pdf, api_k, dane_firmy):
    bezpieczna_nazwa = re.sub(r'[^a-zA-Z0-9_.-]', '_', nazwa_pliku)
    sciezka_cache = os.path.join(KATALOG_CACHE, f"{dane_firmy['nip']}_{bezpieczna_nazwa}.json")
    
    if os.path.exists(sciezka_cache):
        try:
            with open(sciezka_cache, 'r', encoding='utf-8') as f:
                dane_cached = json.load(f)
                dane_cached['nazwa_pliku'] = nazwa_pliku
                return dane_cached
        except Exception:
            pass

    klient = genai.Client(api_key=api_k)
    odp = klient.models.generate_content(
        model="gemini-3.6-flash",
        contents=[types.Part.from_bytes(data=bajty_pdf, mime_type="application/pdf"), buduj_prompt_ekstrakcji(dane_firmy)],
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.1)
    )
    surowe_dane = json.loads(odp.text)
    dane = surowe_dane[0] if isinstance(surowe_dane, list) and len(surowe_dane) > 0 else (surowe_dane if isinstance(surowe_dane, dict) else {})
    dane['nazwa_pliku'] = nazwa_pliku
    
    try:
        with open(sciezka_cache, 'w', encoding='utf-8') as f:
            json.dump(dane, f, ensure_ascii=False, indent=2)
    except Exception:
        pass
    return dane

def czysc_tekst(wartosc):
    if wartosc is None: return ""
    tekst = str(wartosc).strip()
    mapa = {'ł': 'l', 'Ł': 'L', 'ą': 'a', 'Ą': 'A', 'ę': 'e', 'Ę': 'E', 'ć': 'c', 'Ć': 'C', 'ń': 'n', 'Ń': 'N', 'ó': 'o', 'Ó': 'O', 'ś': 's', 'Ś': 'S', 'ź': 'z', 'Ź': 'Z', 'ż': 'z', 'Ż': 'Z', '„': '"', '”': '"', '’': "'", '‘': "'", '–': '-', '—': '-'}
    for k, v in mapa.items(): tekst = tekst.replace(k, v)
    return unicodedata.normalize('NFKD', tekst).encode('ascii', 'ignore').decode('ascii')

def bezpieczny_float(wartosc):
    try:
        if wartosc is None: return 0.0
        if isinstance(wartosc, (int, float)): return round(float(wartosc), 2)
        czysty = str(wartosc).replace(" ", "").replace("PLN", "").replace("zł", "").replace(",", ".")
        znalezione = re.findall(r'-?\d+\.?\d*', czysty)
        if znalezione: return round(float(znalezione[0]), 2)
        return 0.0
    except: return 0.0

def bezpieczny_append(tabela, dane_dict):
    istniejace_pola = {p.upper(): p for p in tabela.field_names}
    oczyszczony = {}
    for k, v in dane_dict.items():
        klucz_up = k.upper()
        if klucz_up not in istniejace_pola: continue
        prawdziwa_nazwa = istniejace_pola[klucz_up]
        info = tabela.field_info(prawdziwa_nazwa)
        limit = info[1] if len(info) > 1 and isinstance(info[1], int) else None
            
        if any(nd in prawdziwa_nazwa.upper() for nd in ['DATA', 'DATAK', 'D_DOK', 'TERMIN']):
            if isinstance(v, str):
                try: v = datetime.date.fromisoformat(czysc_tekst(v)[:10])
                except: v = datetime.date.today()
            oczyszczony[prawdziwa_nazwa] = v if isinstance(v, (datetime.date, datetime.datetime)) else datetime.date.today()
            continue

        typ_str = str(info[0]).upper()
        if 'CHAR' in typ_str or 'MEMO' in typ_str or '67' in typ_str:
            tekst = czysc_tekst(v)
            if limit and isinstance(limit, int): tekst = tekst[:limit]
            oczyszczony[prawdziwa_nazwa] = tekst
        elif 'NUMERIC' in typ_str or 'FLOAT' in typ_str or 'INTEGER' in typ_str or '78' in typ_str:
            oczyszczony[prawdziwa_nazwa] = bezpieczny_float(v)
        elif 'DATE' in typ_str or '68' in typ_str:
            if isinstance(v, str):
                try: v = datetime.date.fromisoformat(czysc_tekst(v)[:10])
                except: v = datetime.date.today()
            oczyszczony[prawdziwa_nazwa] = v if isinstance(v, (datetime.date, datetime.datetime)) else datetime.date.today()
        else:
            tekst = str(v)
            if limit and isinstance(limit, int): tekst = tekst[:limit]
            oczyszczony[prawdziwa_nazwa] = tekst

    try: tabela.append(oczyszczony)
    except Exception as e: raise Exception(f"Błąd zapisu DBF ({os.path.basename(tabela.filename)}): {e}")

def generuj_paczke_dbf_ksp(lista_szablonow, nip_firmy):
    folder_roboczy = 'ksp_export_batch'
    if os.path.exists(folder_roboczy): shutil.rmtree(folder_roboczy)
    os.makedirs(folder_roboczy)

    zrodlo = None
    for kandydat in ['.', '/content', 'wzorzec_paliwo', 'wzorzec_pelny']:
        for root, dirs, files in os.walk(kandydat):
            if 'imp_ks.dbf' in [f.lower() for f in files]:
                zrodlo = root
                break
        if zrodlo: break

    if not zrodlo: raise FileNotFoundError("Brak plików wzorcowych imp_*.dbf!")
    
    for f in os.listdir(zrodlo):
        if f.lower().startswith('imp_') and f.lower().endswith(('.dbf', '.fpt')):
            shutil.copy(os.path.join(zrodlo, f), os.path.join(folder_roboczy, f))

    tabele = {
        'kli': dbf.Table(os.path.join(folder_roboczy, 'imp_kli.dbf'), codepage='cp1250'), 
        'ks': dbf.Table(os.path.join(folder_roboczy, 'imp_ks.dbf'), codepage='cp1250'), 
        'vz': dbf.Table(os.path.join(folder_roboczy, 'imp_vz.dbf'), codepage='cp1250'), 
        'vp': dbf.Table(os.path.join(folder_roboczy, 'imp_vp.dbf'), codepage='cp1250'), 
        'pt': dbf.Table(os.path.join(folder_roboczy, 'imp_pt.dbf'), codepage='cp1250'), 
        'inf': dbf.Table(os.path.join(folder_roboczy, 'imp_inf.dbf'), codepage='cp1250')
    }
    for t in tabele.values(): t.open(mode=dbf.READ_WRITE)
    for klucz in ['kli', 'ks', 'vz', 'vp', 'pt']:
        t = tabele[klucz]
        for rec in t: dbf.delete(rec)
        t.pack()

    licznik_vz, licznik_vp = 0, 0
    for idx, sz_item in enumerate(lista_szablonow, start=1):
        sz = sz_item[0] if isinstance(sz_item, list) and sz_item and isinstance(sz_item[0], dict) else (sz_item if isinstance(sz_item, dict) else {})

        lpp_str = f"{idx:010d}"
        symbol_k = f"K{idx:05d}"
        kierunek = sz.get("kierunek", "ZAKUP")
        kat = sz.get("kategoria_pkpir", "KOLUMNA_13_POZOSTALE")
        
        k_raw = sz.get("kontrahent", {})
        k = k_raw[0] if isinstance(k_raw, list) and k_raw and isinstance(k_raw[0], dict) else (k_raw if isinstance(k_raw, dict) else {})
        
        nip_k = re.sub(r'[^0-9]', '', czysc_tekst(k.get("nip", "") or k.get("tax_id", "")))
        nazwa_k = czysc_tekst(k.get("nazwa", "") or k.get("name", "Brak nazwy"))
        
        ulica_k = czysc_tekst(k.get("ulica") or k.get("street") or "")
        kod_k = czysc_tekst(k.get("kod") or k.get("kod_pocztowy") or k.get("postal_code") or k.get("zip") or "")
        miasto_k = czysc_tekst(k.get("miasto") or k.get("city") or "")
        
        surowy_adres = czysc_tekst(k.get("adres") or k.get("address") or "")
        if surowy_adres and (not ulica_k or not miasto_k):
            match_kod = re.search(r'\d{2}-\d{3}', surowy_adres)
            if match_kod and not kod_k: kod_k = match_kod.group()
            if not ulica_k: ulica_k = surowy_adres[:27]

        pelny_adres = f"{ulica_k}, {kod_k} {miasto_k}".strip(", ")
        if not pelny_adres and surowy_adres: pelny_adres = surowy_adres
        pelny_adres = pelny_adres[:80]

        bezpieczny_append(tabele['kli'], {
            'NAZWA_SK': nazwa_k[:33], 'NAZWA_PL': nazwa_k[:80], 'KOD': kod_k[:6], 
            'MIASTO': miasto_k[:20], 'ULICA': ulica_k[:27], 'NIP': nip_k[:20], 
            'SYMBOL': symbol_k, 'NSYM': idx, 'UE': 'PL', 'TYP': 'D' if kierunek == "ZAKUP" else 'O', 'POLE1': '1'
        })

        kw_raw = sz.get("kwoty", {})
        kw = kw_raw[0] if isinstance(kw_raw, list) and kw_raw and isinstance(kw_raw[0], dict) else (kw_raw if isinstance(kw_raw, dict) else {})

        netto = bezpieczny_float(kw.get("netto"))
        vat_pelny = bezpieczny_float(kw.get("vat"))
        brutto = bezpieczny_float(kw.get("brutto"))
        
        stawka_vat_raw = kw.get("stawka_vat", 23)
        if stawka_vat_raw is None:
            ptu_str = "23"
            s_vat_val = 23.0
        else:
            ptu_str = str(stawka_vat_raw).strip()
            s_vat_val = bezpieczny_float(stawka_vat_raw)

        if netto == 0.0 and brutto > 0.0:
            if s_vat_val > 0:
                netto = round(brutto / (1.0 + (s_vat_val / 100.0)), 2)
                vat_pelny = round(brutto - netto, 2)
            else:
                netto = brutto
                vat_pelny = 0.0
        elif brutto == 0.0 and netto > 0.0:
            vat_pelny = round(netto * (s_vat_val / 100.0), 2)
            brutto = round(netto + vat_pelny, 2)

        try: d_wyst = datetime.date.fromisoformat(czysc_tekst(sz.get("data_wystawienia")))
        except: d_wyst = datetime.date.today()
        try: d_sprz = datetime.date.fromisoformat(czysc_tekst(sz.get("data_sprzedazy")))
        except: d_sprz = d_wyst
        try: d_term = datetime.date.fromisoformat(czysc_tekst(sz.get("termin_platnosci")))
        except: d_term = d_wyst
        
        d_vat = d_wyst

        tdni_val = max(0, (d_term - d_wyst).days)
        nr_dok = czysc_tekst(sz.get("nr_dokumentu", f"FAK_{idx}"))
        opis_zd = czysc_tekst(sz.get("opis_gospodarczy", "Zakup"))
        nr_ksef = czysc_tekst(sz.get("nr_ksef", ""))
        czy_zaplacono = sz.get("czy_zaplacono", False)
        znak_zaplaty = 'T' if czy_zaplacono else 'N'

        rekord_ks = {
            'LPP': lpp_str, 'LP': -1, 'TK': 'Z' if kat == "KOLUMNA_10_TOWARY" else 'I', 
            'TM': 'Z' if kierunek == "ZAKUP" else 'S', 'GRUPA': 'Krajowy', 'NR_DOK': nr_dok, 
            'FIRMA': nazwa_k, 'ADRES': pelny_adres, 'NIP': nip_k, 'SYMBOL': symbol_k, 
            'OPIS': opis_zd[:30], 'DATAK': d_wyst, 'D_DOK': d_sprz, 'DATA_VAT': d_vat, 'TERMIN': d_term, 'TDNI': tdni_val, 
            'DATA_ZAPL': d_wyst, 'ZAPLACONO': znak_zaplaty, 'P_VAT': 1, 'REJ_VAT': 'T', 
            'NR_KSEF': nr_ksef, 'KSEF': nr_ksef, 'ID_KSEF': nr_ksef, 'TYPR': 'F', 'STAN': '1', 'UE': 'PL', 'KSEF_SERW': 'P',
            'KWOTA1': netto, 'KWOTA4': netto
        }
        
        if kat == "SAMOCHOD_MIESZANY":
            v_50 = round(vat_pelny * 0.5, 2)
            v_nie = round(vat_pelny - v_50, 2)
            kup_75 = round((netto + v_nie) * 0.75, 2)
            rekord_ks.update({'KWOTA1': kup_75, 'KWOTA4': v_nie, 'P26': netto, 'P27': v_nie, 'P28': vat_pelny, 'P29': vat_pelny, 'P30': v_50, 'P31': v_50})

        bezpieczny_append(tabele['ks'], rekord_ks)

        rekord_v = {
            'LPP': lpp_str, 'LPP_KS': lpp_str, 'FIRMA': nazwa_k, 'ADRES': pelny_adres, 'NRFAK': nr_dok, 'NIP': nip_k, 'SYMBOL': symbol_k, 
            'DATA': d_vat, 'D_DOK': d_sprz, 'TERMIN': d_term, 'RMC': d_vat.strftime('%Y%m'), 
            'NETTO': netto, 'VAT': vat_pelny, 'BRUTTO': brutto, 'KWOTA': netto, 
            'N1': netto, 'V1': vat_pelny, 'PTU1': ptu_str[:2], 
            'STAN': '1', 'ZAPL': znak_zaplaty, 'ROZLICZ': 'T', 
            'NR_KSEF': nr_ksef, 'KSEF': nr_ksef, 'ID_KSEF': nr_ksef, 'TYPR': 'F', 'UE': 'PL', 'KSEF_SERW': 'P'
        }
        
        if kierunek == "ZAKUP":
            licznik_vz += 1
            if kat == "SAMOCHOD_MIESZANY": 
                rekord_v.update({'VAT': round(vat_pelny*0.5, 2), 'BRUTTO': round(netto + vat_pelny*0.5, 2), 'V1': round(vat_pelny*0.5, 2)})
            bezpieczny_append(tabele['vz'], rekord_v)
        else:
            licznik_vp += 1
            bezpieczny_append(tabele['vp'], rekord_v)

        bezpieczny_append(tabele['pt'], {
            'Z': '1',
            'LPP_G': lpp_str, 
            'LPP': '', 
            'DOK': '1',
            'P1': ptu_str[:2], 
            'KN': netto, 
            'KV': vat_pelny, 
            'KB': brutto, 
            'P2': ptu_str[:2], 
            'KAS': 'N',
            'KOR': 'N',
            'PLIK': 'VZ' if kierunek == "ZAKUP" else 'VP'
        })

    if len(tabele['inf']) > 0:
        with tabele['inf'][0] as r_inf:
            r_inf['DOK'] = len(lista_szablonow)
            r_inf['VZ'] = licznik_vz
            r_inf['VP'] = licznik_vp
            r_inf['DATA'] = datetime.date.today()

    for t in tabele.values(): t.close()
    n_zip = f"KSP_{nip_firmy}_{len(lista_szablonow)}_faktur.zip"
    with zipfile.ZipFile(n_zip, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in os.listdir(folder_roboczy): z.write(os.path.join(folder_roboczy, f), arcname=f)
    return n_zip

pliki_wejsciowe = st.file_uploader(f"📂 Wgraj faktury PDF/ZIP dla: {firma['nazwa']} (PKD: {firma.get('pkd_glowne', 'Brak')})", type=["pdf", "zip"], accept_multiple_files=True)
if pliki_wejsciowe:
    lista_zadan = []
    for el in pliki_wejsciowe:
        if el.name.lower().endswith('.zip'):
            try:
                with zipfile.ZipFile(io.BytesIO(el.getvalue())) as z_in:
                    for w in z_in.namelist():
                        if w.lower().endswith('.pdf') and not w.startswith('__MACOSX'): lista_zadan.append((os.path.basename(w), z_in.read(w)))
            except: pass
        else: lista_zadan.append((el.name, el.getvalue()))

    if st.button(f"🔍 KROK 1: Agent AI uczy się i czyta faktury ({liczba_watkow} wątków)", type="primary"):
        start_t = time.time()
        szablony, pasek, st_text = [], st.progress(0), st.empty()
        with ThreadPoolExecutor(max_workers=liczba_watkow) as executor:
            mapa = {executor.submit(wykonaj_ekstrakcje_pdf, nazwa, bajty, API_KEY, firma): nazwa for nazwa, bajty in lista_zadan}
            zakonczone = 0
            for fut in as_completed(mapa):
                try: 
                    res = fut.result()
                    if res: szablony.append(res)
                except Exception as e: 
                    st.error(f"Błąd: {e}")
                zakonczone += 1
                pasek.progress(zakonczone / len(lista_zadan))
                st_text.text(f"Agent analizuje {zakonczone}/{len(lista_zadan)} faktur...")
        st_text.success(f"✅ Zakończono Krok 1 w {round(time.time() - start_t, 1)} s.")
        st.session_state['szablony_faktur'] = szablony

if 'szablony_faktur' in st.session_state:
    st.markdown("---")
    st.subheader(f"📊 Tabela weryfikacyjna dla agenta firmy: {firma['nazwa']}")
    
    dane_tabeli = []
    for idx, sz_item in enumerate(st.session_state['szablony_faktur']):
        sz = sz_item[0] if isinstance(sz_item, list) and sz_item and isinstance(sz_item[0], dict) else (sz_item if isinstance(sz_item, dict) else {})
        k_raw = sz.get("kontrahent", {})
        k = k_raw[0] if isinstance(k_raw, list) and k_raw and isinstance(k_raw[0], dict) else (k_raw if isinstance(k_raw, dict) else {})
        kw_raw = sz.get("kwoty", {})
        kw = kw_raw[0] if isinstance(kw_raw, list) and kw_raw and isinstance(kw_raw[0], dict) else (kw_raw if isinstance(kw_raw, dict) else {})
        
        dane_tabeli.append({
            "Nr": idx + 1,
            "Plik": sz.get('nazwa_pliku', '-'),
            "Kontrahent": k.get('nazwa', '-'),
            "Kategoria": sz.get('kategoria_pkpir', '-'),
            "Netto": bezpieczny_float(kw.get("netto")),
            "VAT": bezpieczny_float(kw.get("vat")),
            "Brutto": bezpieczny_float(kw.get("brutto")),
            "Stawka": kw.get("stawka_vat", 23),
            "Termin": sz.get('termin_platnosci', '-'),
            "Zapłacono": "Tak" if sz.get('czy_zaplacono') else "Nie"
        })
    
    if dane_tabeli:
        df_wyswietl = pd.DataFrame(dane_tabeli)
        st.dataframe(df_wyswietl, use_container_width=True)

    st.markdown("---")
    if st.button("📦 Wygeneruj gotową paczkę ZIP dla KSP"):
        try:
            zip_plik = generuj_paczke_dbf_ksp(st.session_state['szablony_faktur'], firma['nip'])
            with open(zip_plik, "rb") as fp: st.download_button(label=f"⬇️ Pobierz plik {zip_plik}", data=fp, file_name=zip_plik, mime="application/zip")
        except Exception as e: st.error(f"BŁĄD ZAPISU DBF: {e}")
