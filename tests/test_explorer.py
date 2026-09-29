import json
import threading
import unittest
from urllib.request import urlopen

import pandas as pd

from explorer import LexnetQueries, create_server, split_query


class TestLexnetQueries(unittest.TestCase):
    def setUp(self):
        self.data = {
            'entries': pd.DataFrame([
                {'entry_id': 'e1', 'entry_name': 'chat'},
                {'entry_id': 'e2', 'entry_name': 'prendre le large'},
            ]).set_index('entry_id'),
            'nodes': pd.DataFrame([
                {'node_id': 'n1', 'entry_id': 'e1', 'std_name': 'chat I'},
                {'node_id': 'n2', 'entry_id': 'e1', 'std_name': 'chat II'},
                {'node_id': 'n3', 'entry_id': 'e2', 'std_name': 'prendre le large'},
            ]).set_index('node_id'),
            'forms': pd.DataFrame([
                {'node_id': 'n1', 'features': 'plural', 'signifier': 'chats'},
            ]).set_index('node_id'),
            'features': pd.DataFrame([
                {'node_id': 'n1', 'POS': 'N', 'ph_str': '', 'usage': '', 'features': ['f_noun']},
                {'node_id': 'n3', 'POS': 'V', 'ph_str': 'idiom', 'usage': '', 'features': ['f_idiom']},
                {'node_id': 'n3', 'POS': 'V', 'ph_str': 'idiom', 'usage': '', 'features': ['f_verbal']},
            ]).set_index('node_id'),
            'feature_names': pd.DataFrame([
                {'feature_id': 'f_noun', 'name': 'noun'},
                {'feature_id': 'f_idiom', 'name': 'idiom'},
                {'feature_id': 'f_verbal', 'name': 'verbal expression'},
            ]).set_index('feature_id'),
            'lf_names': pd.DataFrame([
                {'lf_id': 'lf1', 'lf_name': 'Magn', 'type': 'standard'},
                {'lf_id': 'lf2', 'lf_name': 'Oper1', 'type': 'standard'},
            ]).set_index('lf_id'),
            'lfs': pd.DataFrame([
                {'source_id': 'n1', 'lf_id': 'lf1', 'target_id': 'n2', 'form': '', 'frame': '', 'constraint': ''},
                {'source_id': 'n3', 'lf_id': 'lf2', 'target_id': 'n1', 'form': 'prendre', 'frame': '', 'constraint': ''},
            ]),
            'definitions': pd.DataFrame(columns=['node_id', 'def_HTML']).set_index('node_id'),
            'labels': pd.DataFrame(columns=['node_id', 'label']).set_index('node_id'),
            'propforms': pd.DataFrame(columns=['node_id', 'propform']).set_index('node_id'),
            'examples': pd.DataFrame(columns=['ex_id', 'content']).set_index('ex_id'),
            'ex-rel': pd.DataFrame(columns=['node_id', 'ex_id']),
        }
        self.queries = LexnetQueries(self.data)

    def test_split_query_accepts_commas_and_newlines(self):
        self.assertEqual(split_query('Magn, Oper1\nMagn'), ['Magn', 'Oper1'])

    def test_word_search_returns_units_from_matching_entry(self):
        result = self.queries.search_words('chat', mode='exact')
        self.assertEqual(set(result.node_id), {'n1', 'n2'})
        self.assertEqual(set(result.match_source), {'entry'})

    def test_word_search_can_match_an_inflected_form(self):
        result = self.queries.search_words('chats', mode='exact')
        self.assertEqual(result.node_id.tolist(), ['n1'])
        self.assertEqual(result.iloc[0].match_source, 'form')

    def test_lexical_function_search_accepts_a_list(self):
        result = self.queries.search_lexical_functions('Magn, Oper1')
        self.assertEqual(set(result.lf_name), {'Magn', 'Oper1'})
        self.assertEqual(set(result.source_name), {'chat I', 'prendre le large'})

    def test_feature_search_supports_any_and_all(self):
        any_result = self.queries.search_features('idiom, noun')
        all_result = self.queries.search_features('idiom, verbal expression', require_all=True)
        self.assertEqual(set(any_result.node_id), {'n1', 'n3'})
        self.assertEqual(all_result.node_id.tolist(), ['n3'])

    def test_node_description_resolves_feature_names(self):
        description = self.queries.describe_node('n3')
        self.assertIn('prendre le large', description)
        self.assertIn('Features: idiom', description)
        self.assertIn('Oper1', description)

    def test_lexical_function_links_target_related_units(self):
        outgoing = self.queries.lexical_function_links('n3')
        incoming = self.queries.lexical_function_links('n1')
        self.assertEqual(outgoing[0]['node_id'], 'n1')
        self.assertEqual(outgoing[0]['node_name'], 'chat I')
        self.assertEqual({link['node_id'] for link in incoming}, {'n2', 'n3'})

    def test_local_server_serves_page_and_search_api(self):
        server = create_server(self.queries, '/tmp/example')
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        base_url = f'http://127.0.0.1:{server.server_port}'
        try:
            with urlopen(base_url + '/') as response:
                page = response.read().decode('utf-8')
            with urlopen(base_url + '/api/search?kind=word&q=chat&mode=exact&forms=1') as response:
                payload = json.load(response)
            with urlopen(base_url + '/api/node?id=n3') as response:
                node_payload = json.load(response)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

        self.assertIn('LexNet Explorer', page)
        self.assertIn('function sortBy(key)', page)
        self.assertIn("th.setAttribute('aria-sort'", page)
        self.assertIn("key === 'source_name' || key === 'target_name' || (key === 'form' && row.form)", page)
        self.assertIn("row.source_id : row.target_id", page)
        self.assertIn("(kind === 'word' || kind === 'feature') && key === 'std_name'", page)
        self.assertEqual({row['node_id'] for row in payload['rows']}, {'n1', 'n2'})
        self.assertEqual(node_payload['lexical_function_links'][0]['node_id'], 'n1')


if __name__ == '__main__':
    unittest.main()
