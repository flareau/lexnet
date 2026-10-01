import unittest
from unittest.mock import patch

import pandas as pd
import models


class TestModels(unittest.TestCase):
    def setUp(self):
        self.ln = {
            'nodes': pd.DataFrame(
                [
                    {'node_id': 'n1', 'entry_id': 'e1', 'lexname': 'alpha', 'std_name': 'alpha'},
                    {'node_id': 'n2', 'entry_id': 'e1', 'lexname': 'beta', 'std_name': 'beta'},
                    {'node_id': 'n3', 'entry_id': 'e2', 'lexname': 'gamma', 'std_name': 'gamma'},
                ]
            ).set_index('node_id'),
            'entries': pd.DataFrame(
                [
                    {'entry_id': 'e1', 'entry_name': 'entry1', 'entry_status': 'ok'},
                    {'entry_id': 'e2', 'entry_name': 'entry2', 'entry_status': 'ok'},
                ]
            ).set_index('entry_id'),
            'copolysemy': pd.DataFrame([
                {'source_node_id': 'n1', 'target_node_id': 'n2', 'cp_type': 'ct1', 'cp_subtype': 'cs1'},
            ]),
            'copolysemy_types': pd.DataFrame([
                {'cp_type': 'ct1', 'name': 'Metaphor', 'order': 1, 'semantics': 2, 'derivation': True},
            ]).set_index('cp_type'),
            'copolysemy_subtypes': pd.DataFrame([
                {'cp_subtype': 'cs1', 'cp_type': 'ct1', 'name': 'Form'},
            ]).set_index('cp_subtype'),
            'features': pd.DataFrame(
                [
                    {'node_id': 'n1', 'usage': '', 'usagevars': '', 'POS': 'N', 'ph_str': '', 'embeddedlex': '', 'features': ['f1', 'f2'], 'featuresvars': ''},
                    {'node_id': 'n2', 'usage': '', 'usagevars': '', 'POS': 'V', 'ph_str': '', 'embeddedlex': '', 'features': ['f3'], 'featuresvars': ''},
                ]
            ).set_index('node_id'),
            'forms': pd.DataFrame(
                [
                    {'node_id': 'n1', 'features': '(plural)', 'signifier': 'alphas'},
                    {'node_id': 'n1', 'features': '(singular)', 'signifier': 'alpha'},
                    {'node_id': 'n2', 'features': '(present)', 'signifier': 'beta'},
                ]
            ).set_index('node_id'),
            'feature_names': pd.DataFrame(
                [
                    {'feature_id': 'gf1', 'name': 'POS'},
                    {'feature_id': 'gf2', 'name': 'gender'},
                ]
            ).set_index('feature_id'),
            'label_names': pd.DataFrame(
                [
                    {'semantic_label_id': 'sl1', 'name': 'Label One'},
                    {'semantic_label_id': 'sl2', 'name': 'Label Two'},
                ]
            ).set_index('semantic_label_id'),
            'labels': pd.DataFrame(
                [
                    {'node_id': 'n1', 'semantic_label_id': 'sl1', 'label_%': 100},
                    {'node_id': 'n2', 'semantic_label_id': 'sl2', 'label_%': 100},
                ]
            ).set_index('node_id'),
            'propforms': pd.DataFrame(
                [
                    {'node_id': 'n1', 'propform': 'X does Y', 'tildevalue': '', 'propform_confid': 100, 'actants': ''},
                    {'node_id': 'n2', 'propform': 'X has Y', 'tildevalue': '', 'propform_confid': 100, 'actants': ''},
                ]
            ).set_index('node_id'),
            'definitions': pd.DataFrame(
                [
                    {'node_id': 'n1', 'def_XML': '<def>1</def>', 'def_HTML': '<p>def 1</p>'},
                    {'node_id': 'n2', 'def_XML': '<def>2</def>', 'def_HTML': '<p>def 2</p>'},
                ]
            ).set_index('node_id'),
            'lf_names': pd.DataFrame(
                [
                    {'lexical_function_id': 'lf1', 'lf_name': 'Magn', 'type': 'standard'},
                    {'lexical_function_id': 'lf2', 'lf_name': 'Oper1', 'type': 'standard'},
                ]
            ).set_index('lexical_function_id'),
            'examples': pd.DataFrame(
                [
                    {'example_id': 'x1', 'source': 'src', 'status': 'ok', 'content': 'example 1', 'title': '', 'authors': '', 'location': '', 'date': ''},
                    {'example_id': 'x2', 'source': 'src', 'status': 'ok', 'content': 'example 2', 'title': '', 'authors': '', 'location': '', 'date': ''},
                ]
            ).set_index('example_id'),
            'ex-rel': pd.DataFrame(
                [
                    {'node_id': 'n1', 'example_id': 'x1', 'occurrence': 1, 'position': 1, '%': 100},
                    {'node_id': 'n1', 'example_id': 'x2', 'occurrence': 1, 'position': 2, '%': 100},
                    {'node_id': 'n2', 'example_id': 'x2', 'occurrence': 1, 'position': 1, '%': 100},
                ]
            ),
        }

    def test_lexical_network_calls_loader(self):
        with patch('models.load', return_value=self.ln) as mock_load:
            network = models.LexicalNetwork(path='/tmp/data', separator=';', encoding='latin1')

        mock_load.assert_called_once_with(path='/tmp/data', separator=';', encoding='latin1')
        self.assertEqual(network.path, '/tmp/data')
        self.assertEqual(network.separator, ';')
        self.assertEqual(network.encoding, 'latin1')
        self.assertIs(network.data, self.ln)
        self.assertIs(network['nodes'], self.ln['nodes'])

        unit = models.LexicalUnit(node_id='n1', ln=network)
        self.assertEqual(unit.data['lexname'], 'alpha')

    def test_lexical_unit_selects_node_and_features(self):
        unit = models.LexicalUnit(node_id='n1', ln=self.ln)

        self.assertEqual(unit.id, 'n1')
        self.assertEqual(unit.data['lexname'], 'alpha')
        self.assertEqual(unit.features['POS'], 'N')
        self.assertEqual(unit.features['features'], ['f1', 'f2'])
        self.assertEqual(unit.wordforms['signifier'].tolist(), ['alphas', 'alpha'])
        self.assertEqual(set(unit.examples.index), {'x1', 'x2'})
        self.assertEqual(unit.outgoing_copolysemy['target_node_id'].tolist(), ['n2'])
        self.assertTrue(unit.incoming_copolysemy.empty)

    def test_lexical_entry_selects_entry_and_senses(self):
        entry = models.LexicalEntry(entry_id='e1', ln=self.ln)

        self.assertEqual(entry.id, 'e1')
        self.assertEqual(entry.data['entry_name'], 'entry1')
        self.assertEqual(set(entry.senses.index.tolist()), {'n1', 'n2'})
        self.assertEqual(entry.copolysemy['target_node_id'].tolist(), ['n2'])

    def test_copolysemy_type_selects_metadata_and_subtypes(self):
        cp_type = models.CopolysemyType(cp_type='ct1', ln=self.ln)

        self.assertEqual(cp_type.id, 'ct1')
        self.assertEqual(cp_type.data['name'], 'Metaphor')
        self.assertEqual(cp_type.subtypes.index.tolist(), ['cs1'])

    def test_copolysemy_subtype_selects_metadata(self):
        subtype = models.CopolysemySubtype(cp_subtype='cs1', ln=self.ln)

        self.assertEqual(subtype.id, 'cs1')
        self.assertEqual(subtype.data['name'], 'Form')

    def test_grammatical_feature_selects_feature_name(self):
        feature = models.GrammaticalFeature(feature_id='gf2', ln=self.ln)

        self.assertEqual(feature.id, 'gf2')
        self.assertEqual(feature.data['name'], 'gender')

    def test_semantic_label_selects_label_name(self):
        label = models.SemanticLabel(semantic_label_id='sl1', ln=self.ln)

        self.assertEqual(label.id, 'sl1')
        self.assertEqual(label.data['name'], 'Label One')

    def test_propositional_form_selects_form(self):
        prop = models.PropositionalForm(node_id='n2', ln=self.ln)

        self.assertEqual(prop.id, 'n2')
        self.assertEqual(prop.data['propform'], 'X has Y')

    def test_definition_selects_definition_by_node_id(self):
        definition = models.Definition(node_id='n2', ln=self.ln)

        self.assertEqual(definition.id, 'n2')
        self.assertEqual(definition.data['def_HTML'], '<p>def 2</p>')

    def test_lexical_function_selects_lf_name(self):
        lf = models.LexicalFunction(lexical_function_id='lf1', ln=self.ln)

        self.assertEqual(lf.id, 'lf1')
        self.assertEqual(lf.data['lf_name'], 'Magn')

    def test_example_selects_example_by_id(self):
        example = models.Example(example_id='x1', ln=self.ln)

        self.assertEqual(example.id, 'x1')
        self.assertEqual(example.data['content'], 'example 1')


if __name__ == '__main__':
    unittest.main()
