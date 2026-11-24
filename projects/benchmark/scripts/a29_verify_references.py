"""A29-13: verify every bibliography entry against Crossref (DOI), arXiv (eprint) or the URL itself.

Prints and writes results/a29/references/reference_check.json. Generic User-Agent, no personal data.
  python a29_verify_references.py path/to/references.bib
"""
import difflib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from a29_common import RESULTS, dump

UA = {'User-Agent': 'reference-checker/1.0 (academic manuscript QA)'}


def get(url, timeout=30):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()


def norm(t):
    return re.sub(r'[^a-z0-9 ]', '', re.sub(r'[{}\\]', '', t.lower()))


def entries(text):
    """One chunk per '@type{key,' so that an entry ending in '}}' cannot swallow the next one."""
    for chunk in re.split(r'\n(?=@\w+\s*\{)', '\n' + text):
        m = re.match(r'\s*@(\w+)\s*\{\s*([^,\s]+)\s*,(.*)', chunk, re.S)
        if not m or m.group(1).lower() in ('comment', 'string', 'preamble'):
            continue
        body = m.group(3)
        f = {}
        for k, v in re.findall(r'(\w+)\s*=\s*\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}', body):
            f.setdefault(k.lower(), v.strip())
        yield m.group(2).strip(), f


def main(bib):
    text = open(bib).read()
    out = []
    for key, f in entries(text):
        rec = {'key': key, 'title': f.get('title', '')}
        try:
            if 'doi' in f:
                st, body = get('https://api.crossref.org/works/' + urllib.parse.quote(f['doi']))
                msg = json.loads(body)['message']
                found = (msg.get('title') or [''])[0]
                rec.update(source='crossref', doi=f['doi'], found_title=found, container=(msg.get('container-title') or [''])[0],
                           year=(msg.get('issued', {}).get('date-parts') or [[None]])[0][0])
            elif f.get('eprint'):
                st, body = get('http://export.arxiv.org/api/query?id_list=' + f['eprint'])
                ns = {'a': 'http://www.w3.org/2005/Atom'}
                e = ET.fromstring(body).find('a:entry', ns)
                found = ' '.join(e.find('a:title', ns).text.split()) if e is not None else ''
                rec.update(source='arxiv', eprint=f['eprint'], found_title=found)
            elif 'url' in f:
                st, body = get(f['url'])
                rec.update(source='url', url=f['url'], http_status=st)
                found = f.get('title', '') if st == 200 else ''
            else:
                rec.update(source='none'); found = ''
            rec['title_similarity'] = round(difflib.SequenceMatcher(None, norm(rec['title']), norm(found)).ratio(), 3) if found else None
            rec['status'] = 'OK' if (rec.get('title_similarity') or 0) >= 0.85 or (rec['source'] == 'url' and rec.get('http_status') == 200) else 'CHECK'
        except Exception as e:
            rec.update(status='ERROR', error=repr(e)[:200])
        out.append(rec)
        print(rec['status'], key, rec.get('title_similarity'), '|', rec.get('found_title', rec.get('error', ''))[:90], flush=True)
        time.sleep(0.3)
    (RESULTS / 'references').mkdir(parents=True, exist_ok=True)
    dump(RESULTS / 'references' / 'reference_check.json', out)


if __name__ == '__main__':
    main(sys.argv[1])
