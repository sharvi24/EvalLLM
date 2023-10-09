import unittest
from EvalLLM import PromptCreator


class TestPromptCreator(unittest.TestCase):
    def setUp(self):
        self.prompt_creator = PromptCreator()

    def test_create_prompts(self):
        data_location = 'tests/test_data'
        prompts = self.prompt_creator.create_prompts(data_location)
        self.assertIsInstance(prompts, dict)
        self.assertIn('story1', prompts)
        self.assertIn('prompts', prompts['story1'])
        self.assertIn('targets', prompts['story1'])

    def test_create_prompts_with_no_data(self):
        data_location = 'empty_data'
        prompts = self.prompt_creator.create_prompts(data_location)
        self.assertEqual(prompts, {})

    def test_get_story_names(self):
        data_location = 'tests/test_data'
        self.prompt_creator = PromptCreator()
        story_names = self.prompt_creator.get_story_names(data_location)

        self.assertEqual(len(story_names), 1)
        self.assertIn('story1', story_names)


if __name__ == '__main__':
    unittest.main()
