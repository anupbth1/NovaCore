"""tools/solver.py — Pure-NovaCore deterministic precision solver.

EXACT, rule-based, instant. Turns numeric word questions into exact answers —
the domain where current LLMs make arithmetic mistakes and NovaCore never does.
Every solver is a closed-form rule (no weights, no data lookup, no LM).

Wired into the CDE loop as the fastest priority path: if a query matches a
solver pattern, the exact answer is returned immediately with route='solver'.
Return values are Python floats/ints/str — never a guessed sentence.
"""

import re
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# SI unit tables (exact factors, base unit first)
# ---------------------------------------------------------------------------
_LENGTH = {'m': 1.0, 'meter': 1.0, 'meters': 1.0, 'metre': 1.0, 'metres': 1.0,
           'km': 1000.0, 'kilometer': 1000.0, 'kilometers': 1000.0,
           'cm': 0.01, 'centimeter': 0.01, 'centimeters': 0.01,
           'mm': 0.001, 'millimeter': 0.001, 'millimeters': 0.001,
           'ft': 0.3048, 'foot': 0.3048, 'feet': 0.3048,
           'in': 0.0254, 'inch': 0.0254, 'inches': 0.0254,
           'yd': 0.9144, 'yard': 0.9144, 'yards': 0.9144,
           'mi': 1609.344, 'mile': 1609.344, 'miles': 1609.344}
_MASS = {'kg': 1.0, 'kilogram': 1.0, 'kilograms': 1.0, 'kilo': 1.0,
         'g': 0.001, 'gram': 0.001, 'grams': 0.001,
         'mg': 1e-6, 'milligram': 1e-6, 'milligrams': 1e-6,
         'lb': 0.45359237, 'lbs': 0.45359237, 'pound': 0.45359237,
         'pounds': 0.45359237, 'oz': 0.028349523125, 'ounce': 0.028349523125,
         'ounces': 0.028349523125, 'ton': 907.18474, 'tons': 907.18474,
         't': 1000.0, 'tonne': 1000.0, 'tonnes': 1000.0}
_TIME = {'s': 1.0, 'sec': 1.0, 'second': 1.0, 'seconds': 1.0,
         'min': 60.0, 'minute': 60.0, 'minutes': 60.0,
         'hr': 3600.0, 'hour': 3600.0, 'hours': 3600.0,
         'day': 86400.0, 'days': 86400.0,
         'week': 604800.0, 'weeks': 604800.0}
_DATA = {'b': 1.0, 'byte': 1.0, 'bytes': 1.0,
         'kb': 1000.0, 'kilobyte': 1000.0, 'kilobytes': 1000.0,
         'mb': 1e6, 'megabyte': 1e6, 'megabytes': 1e6,
         'gb': 1e9, 'gigabyte': 1e9, 'gigabytes': 1e9,
         'tb': 1e12, 'terabyte': 1e12, 'terabytes': 1e12,
         'kib': 1024.0, 'mib': 2 ** 20, 'gib': 2 ** 30}


class Solver:
    """Deterministic numeric/unit word-problem solvers (pure rules)."""

    def solve(self, query):
        """Return an exact short answer (str) or None if no solver matched."""
        if not query or not isinstance(query, str):
            return None
        q = query.strip()
        got = None
        for fn in (self._wordproblem, self._transitive, self._temperature,
                   self._percent, self._unit_conv, self._speed, self._date,
                   self._area):
            got = fn(q)
            if got is not None:
                break
        return got

    # ------------------------------------------------------------------
    # Multi-number STORY PROBLEMS (exact, closed-form)
    # ------------------------------------------------------------------
    def _wordproblem(self, q):
        ql = q.lower()
        # Generic discount/pay: "<...> B% off/discount <...base...>" with a
        # pay/final/price trigger -> final = base - discount.
        m = re.search(r'(\d+(?:\.\d+)?)\s*%\s*(?:off|discount)', ql)
        if m and re.search(r'(?:pay|price|cost\w*|amount|final|much|buy)', ql):
            nums = [self._pt(x) for x in re.findall(r'\d+(?:\.\d+)?', ql)]
            pct = self._pt(m.group(1))
            base = None
            for n in nums:
                if abs(n - pct) > 1e-9:
                    base = n
            if base is not None:
                return self._fmt(round(base * (1 - pct / 100.0), 10))
        # "<qty> [items/kg…] at <price> each/per" -> total = qty*price
        if re.search(r'(?:cost|total|much|buy|each|per)', ql):
            m = re.search(r'(\d+(?:\.\d+)?)\s+(?:items?|pieces?|kg|kg\b|units?)?'
                          r'\s*at\s+(\d+(?:\.\d+)?)\s+(?:per|each)', ql)
            if m:
                return self._fmt(self._pt(m.group(1)) * self._pt(m.group(2)))
        # sum/difference with one quantity word ("5 plus 3", "10 minus 4")
        s = ql
        for w, op in (('multiplied by', '*'), ('times', '*'),
                      ('divided by', '/'), ('plus', '+'), ('minus', '-'),
                      ('and', '+')):
            s = s.replace(' ' + w + ' ', ' ' + op + ' ')
        m = re.search(r'(-?\d+(?:\.\d+)?\s*[+\-*/]\s*\d+(?:\.\d+)?)', s)
        if m:
            return self._fmt(self._safe_arith(m.group(1).replace(' ', '')))
        return None

    # -- transitive relational reasoning ---------------------------------
    _ATTR = {
        # pole word -> (attribute, sign); sign +1 = MORE of attribute
        'taller': ('tall', 1), 'tallest': ('tall', 1), 'tall': ('tall', 1),
        'shorter': ('tall', -1), 'shortest': ('tall', -1),
        'older': ('old', 1), 'oldest': ('old', 1),
        'younger': ('old', -1), 'youngest': ('old', -1),
        'heavier': ('heavy', 1), 'heaviest': ('heavy', 1),
        'lighter': ('heavy', -1), 'lightest': ('heavy', -1),
        'bigger': ('big', 1), 'biggest': ('big', 1),
        'smaller': ('big', -1), 'smallest': ('big', -1),
        'faster': ('fast', 1), 'fastest': ('fast', 1),
        'slower': ('fast', -1), 'slowest': ('fast', -1),
        'higher': ('high', 1), 'highest': ('high', 1),
        'lower': ('low', 1), 'lowest': ('low', -1),
        'richer': ('rich', 1), 'richest': ('rich', 1),
        'stronger': ('strong', 1), 'strongest': ('strong', 1),
        'weaker': ('strong', -1), 'weakest': ('strong', -1),
        'longer': ('long', 1), 'longest': ('long', 1),
        'hotter': ('hot', 1), 'hottest': ('hot', 1),
        'colder': ('cold', 1), 'coldest': ('cold', 1),
        'older than': ('old', 1),
    }
    _ENT_STOP = {'who', 'what', 'which', 'whom', 'where', 'when'}

    def _transitive(self, q):
        ql = q.lower()
        if 'than' not in ql:
            return None
        edges = {}  # attr -> set of (greater, lesser)
        for m in re.finditer(
                r'([a-z][a-z\s-]{0,15}?)\s+is\s+'
                r'(taller|tallest|shorter|shortest|older|oldest|younger|'
                r'youngest|heavier|heaviest|lighter|lightest|bigger|biggest|'
                r'smaller|smallest|faster|fastest|slower|slowest|higher|highest|'
                r'lower|lowest|richer|richest|stronger|strongest|weaker|weakest|'
                r'longer|longest|hotter|hottest|colder|coldest)'
                r'\s+than\s+([a-z][a-z\s-]{0,15}?)(?=[\s,.;!?]|$)',
                ql):
            a, pole, b = m.group(1).strip(), m.group(2), m.group(3).strip()
            if not a or not b:
                continue
            if a.lower() in self._ENT_STOP or b.lower() in self._ENT_STOP:
                continue
            attr, sign = self._ATTR[pole]
            edges.setdefault(attr, set()).add((a, b) if sign > 0 else (b, a))
        if not edges:
            return None
        # Attribute to optimise: prefer the pole asked in the QUESTION. Fact
        # poles are the ones inside "X is <pole> than Y"; the question pole
        # is any pole word that is NOT part of a fact clause.
        attr = next(iter(edges))
        want_sign = 1
        fact_poles = set()
        for m in re.finditer(
                r'\s+is\s+(taller|tallest|shorter|shortest|older|oldest|'
                r'younger|youngest|heavier|heaviest|lighter|lightest|bigger|'
                r'biggest|smaller|smallest|faster|fastest|slower|slowest|'
                r'higher|highest|lower|lowest|richer|richest|stronger|'
                r'strongest|weaker|weakest|longer|longest|hotter|hottest|'
                r'colder|coldest)\s+than', ql):
            fact_poles.add(m.group(1))
        poles = re.findall(r'(tallest|taller|shortest|older|oldest|younger|'
                           r'youngest|heaviest|heavier|lightest|biggest|bigger|'
                           r'smallest|fastest|faster|slowest|highest|higher|'
                           r'lowest|richest|strongest|weakest|longest|hottest|'
                           r'coldest)', ql)
        q_pole = next((p for p in poles if p not in fact_poles), None)
        if q_pole is None:
            # no explicit question pole: ambiguous facts only
            unique = {p for p in poles}
            q_pole = next(iter(unique)) if len(unique) == 1 else None
        if q_pole and q_pole in self._ATTR:
            attr, want_sign = self._ATTR[q_pole]
        es = edges.get(attr)
        if not es:
            return None
        return self._extreme(es, want_sign)

    def _extreme(self, es, want_sign):
        greater = {a for a, b in es}
        lesser = {b for a, b in es}
        if want_sign > 0:
            cands = [c for c in (greater - lesser) if c]
        else:
            cands = [c for c in (lesser - greater) if c]
        if len(cands) == 1 and cands[0].lower() not in self._ENT_STOP:
            return cands[0].strip()
        return None

    # ------------------------------------------------------------------
    @staticmethod
    def _safe_arith(expr):
        """Evaluate a tiny arithmetic expression (numbers + - * /) with ast —
        never eval()."""
        import ast
        def _ev(node):
            if isinstance(node, ast.Expression):
                return _ev(node.body)
            if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                return node.value
            if isinstance(node, ast.BinOp):
                l, r = _ev(node.left), _ev(node.right)
                from operator import add, sub, mul, truediv
                ops = {ast.Add: add, ast.Sub: sub, ast.Mult: mul, ast.Div: truediv}
                for t, op in ops.items():
                    if isinstance(node.op, t):
                        return op(l, r)
            if isinstance(node, ast.UnaryOp):
                from operator import neg, pos
                if isinstance(node.op, ast.USub):
                    return -_ev(node.operand)
                if isinstance(node.op, ast.UAdd):
                    return +_ev(node.operand)
            raise ValueError("unsupported")
        try:
            tree = ast.parse(expr, mode='eval')
            allowed = (ast.Expression, ast.Constant, ast.BinOp, ast.UnaryOp)
            for node in ast.walk(tree):
                if isinstance(node, ast.operator):
                    if not isinstance(node, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
                        raise ValueError("bad op")
                    continue
                if not isinstance(node, allowed):
                    raise ValueError("bad node")
            val = _ev(tree.body)
            if isinstance(val, float) and abs(val - round(val)) < 1e-10:
                val = int(round(val))
            return val
        except Exception:
            return None

    # ------------------------------------------------------------------
    @staticmethod
    def _pt(s):
        return float(s.replace(',', ''))

    # -- unit conversion ------------------------------------------------
    def _unit_conv(self, q):
        aliases = {'kilometres': 'kilometers', 'metres': 'meters',
                   'centimetres': 'centimeters', 'millimetres': 'millimeters'}
        ql = q.lower()
        for a, b in aliases.items():
            ql = ql.replace(a, b)
        un = ('km|kilometers?|meters?|metres?|centimeters?|cm|millimeters?|'
              'mm|feet|foot|ft|inches?|in|miles?|mi|yards?|yd|kilograms?|'
              'kilo|kg|grams?|g|milligrams?|mg|pounds?|lbs?|ounces?|oz|'
              'tons?|tonnes?|seconds?|s\\b|minutes?|min\\b|hours?|hr\\b|'
              'days?|weeks?|bytes?|b\\b|kilobytes?|kb\\b|megabytes?|mb\\b|'
              'gigabytes?|gb\\b|terabytes?|tb\\b|kib\\b|mib\\b|gib\\b')
        ul = '(?:' + un + r')\b'
        # "convert X <u1> to <u2>"
        m = re.search(r'(\d+(?:\.\d+)?)\s*(' + un + r')\b'
                      r'[^a-z0-9]{0,5}?to[^a-z0-9]{0,5}(' + un + r')\b', ql)
        if m:
            value, u1 = self._pt(m.group(1)), m.group(2).strip().rstrip('.')
            u2 = m.group(3).strip().rstrip('.')
            table = next((t for t in (_LENGTH, _MASS, _TIME, _DATA)
                          if u1 in t and u2 in t), None)
            if table is not None:
                return self._fmt(value * table[u1] / table[u2], u2)
        # "how many <target> in <value> <source>" (reversed order)
        m2 = re.search(r'how\s+many\s+(' + un + r')\b\s+in\s+'
                       r'(\d+(?:\.\d+)?)\s*(' + un + r')\b', ql)
        if m2:
            u_target = m2.group(1).rstrip('s')
            value = self._pt(m2.group(2))
            u_src = m2.group(3).rstrip('s')
            table = next((t for t in (_LENGTH, _MASS, _TIME)
                          if u_target in t and u_src in t), None)
            if table is not None:
                return self._fmt(value * table[u_src] / table[u_target],
                                 u_target)
        # "total meters when X km and Y cm" style is out of scope
        return None

    # -- percent ----------------------------------------------------------
    def _percent(self, q):
        ql = q.lower()
        m = re.search(r'(?:what\s+is\s+)?(\d+(?:\.\d+)?)\s*(?:%|percent)\s*'
                      r'(?:off|of|discount on)\s+(\d+(?:\.\d+)?)', ql)
        if m:
            pct, base = self._pt(m.group(1)), self._pt(m.group(2))
            return self._fmt(base * pct / 100.0)
        # "what is 40 percent of 75" handled above; now "x% of y" with/without is
        # "what percent of 50 is 10" -> 10/50 = 20%
        m = re.search(r'what\s+(?:percentage|percent|%)\s+of\s+(\d+(?:\.\d+)?)'
                      r'\s+is\s+(\d+(?:\.\d+)?)', ql)
        if m:
            b, a = self._pt(m.group(1)), self._pt(m.group(2))
            if b == 0:
                return None
            return self._fmt(a / b * 100.0) + '%'
        # "what is 10 as a percent of 50"
        m = re.search(r'(\d+(?:\.\d+)?)\s*(?:as\s+a\s+)?percent\s+of\s+'
                      r'(\d+(?:\.\d+)?)', ql)
        if m and 'what percent' not in ql:
            a, b = self._pt(m.group(1)), self._pt(m.group(2))
            if b == 0:
                return None
            return self._fmt(a / b * 100.0) + '%'
        m = re.search(r'increase\s+(\d+(?:\.\d+)?)\s+by\s+(\d+(?:\.\d+)?)\s*'
                      r'(?:%|percent)', ql)
        if m:
            base, pct = self._pt(m.group(1)), self._pt(m.group(2))
            return self._fmt(base * (1 + pct / 100.0))
        m = re.search(r'(\d+(?:\.\d+)?)\s*(?:is|equal to)\s+what\s+'
                      r'(?:percentage|percent|%)\s+of\s+(\d+(?:\.\d+)?)', ql)
        if m:
            a, b = self._pt(m.group(1)), self._pt(m.group(2))
            if b == 0:
                return None
            return self._fmt(a / b * 100.0) + '%'
        return None

    # -- speed (distance / time -> speed in km/h or m/s) ------------------
    def _speed(self, q):
        ql = q.lower()
        if not ('speed' in ql or 'velocity' in ql):
            return None
        d = re.search(r'(?:distance|travel|travels|covers|covering|covers?|moves)\s*(?:of|is)?\s*'
                      r'(\d+(?:\.\d+)?)\s*(km|m|meters?|kilometers?)', ql)
        t = re.search(r'(?:time|took|takes)?\s*(\d+(?:\.\d+)?)\s*'
                      r'(hours?|hr|minutes?|seconds?|s\b)', ql)
        if not d or not t:
            return None
        dist = self._pt(d.group(1))
        is_km = d.group(2).startswith(('km', 'kilometer'))
        tval = self._pt(t.group(1))
        tunit = t.group(2)
        if tunit in ('s', 'sec', 'second', 'seconds'):
            hours = tval / 3600.0
            isn_s = True
        elif tunit.startswith(('min', 'minute')):
            hours = tval / 60.0
            isn_s = False
        else:
            hours = tval
            isn_s = False
        if is_km:
            return self._fmt(dist / hours, 'km/h')
        if isn_s:
            return self._fmt(dist / tval, 'm/s')
        return self._fmt(dist / hours, 'm/h')

    # -- temperature ------------------------------------------------------
    def _temperature(self, q):
        ql = q.lower()
        m = re.search(r'(\d+(?:\.\d+)?)\s*°?\s*([cfk]|celsius|fahrenheit|kelvin)'
                      r'\s*to\s*([cfk]|celsius|fahrenheit|kelvin)', ql)
        if not m:
            return None
        val = self._pt(m.group(1))
        a, b = m.group(2)[0].lower(), m.group(3)[0].lower()
        if a == b or a not in 'cfk' or b not in 'cfk':
            return None
        if a == 'c':
            cel = val
        elif a == 'f':
            cel = (val - 32) * 5 / 9
        else:
            cel = val - 273.15
        if b == 'c':
            out = cel
        elif b == 'f':
            out = cel * 9 / 5 + 32
        else:
            out = cel + 273.15
        return self._fmt(out, 'C' if b == 'c' else ('F' if b == 'f' else 'K'))

    # -- dates -------------------------------------------------------------
    def _date(self, q):
        ql = q.lower()
        m = re.search(r'days?\s+between\s+([a-z0-9]+)\s+([0-9]+)[,\s-]*([0-9]{2,4})'
                      r'\s+and\s+([a-z0-9]+)\s+([0-9]+)[,\s-]*([0-9]{2,4})', ql)
        if m:
            try:
                d1 = self._parse_date(m.group(1), m.group(2), m.group(3))
                d2 = self._parse_date(m.group(4), m.group(5), m.group(6))
                return f"{(d2 - d1).days} days"
            except Exception:
                return None
        m = re.search(r'what\s+date\s+(?:is|will\s+be)\s+([0-9]+)\s+days?\s+after\s+'
                      r'([a-z0-9]+)\s+([0-9]+)[,\s-]*([0-9]{2,4})', ql)
        if m:
            try:
                d1 = self._parse_date(m.group(2), m.group(3), m.group(4))
                return (d1 + timedelta(days=int(m.group(1)))).strftime('%Y-%m-%d')
            except Exception:
                return None
        return None

    @staticmethod
    def _parse_date(mon, day, year):
        year = int(year)
        if year < 100:
            year += 2000
        months = {'jan': 1, 'january': 1, 'feb': 2, 'february': 2, 'mar': 3,
                  'march': 3, 'apr': 4, 'april': 4, 'may': 5, 'jun': 6,
                  'june': 6, 'jul': 7, 'july': 7, 'aug': 8, 'august': 8,
                  'sep': 9, 'sept': 9, 'september': 9, 'oct': 10,
                  'october': 10, 'nov': 11, 'november': 11, 'dec': 12,
                  'december': 12}
        m = months.get(mon[:3].lower())
        if m is None:
            raise ValueError(mon)
        return datetime(year, m, int(day))

    # -- area of common shapes ---------------------------------------------
    def _area(self, q):
        ql = q.lower()
        if 'area' not in ql:
            return None
        r = re.search(r'(?:radius|r)\s*(?:of|is|=)?\s*(\d+(?:\.\d+)?)', ql)
        c = re.search(r'circle', ql)
        if c and r:
            import math
            rad = self._pt(r.group(1))
            return self._fmt(math.pi * rad * rad, 'sq units')
        if 'rectangle' in ql or 'square' in ql:
            a = re.search(r'(\d+(?:\.\d+)?)\s*[x×]\s*(\d+(?:\.\d+)?)', ql)
            if a:
                return self._fmt(self._pt(a.group(1)) * self._pt(a.group(2)),
                                 'sq units')
        return None

    # ------------------------------------------------------------------
    @staticmethod
    def _fmt(value, unit=''):
        if isinstance(value, float) and abs(value - round(value)) < 1e-10:
            value = int(round(value))
        from decimal import Decimal
        if isinstance(value, float):
            value = float(Decimal(str(round(value, 6))))
        u = ' ' + unit if unit else ''
        return f"{value}{u}"