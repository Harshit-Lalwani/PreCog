import json
from pathlib import Path
from typing import Dict, List

class PromptManager:
    def __init__(self, prompts_file: Path = Path("/root/PreCog/src/prompts.json")):
        self.prompts_file = prompts_file
        self._load_prompts()
    
    def _load_prompts(self):
        """Load prompts from JSON file"""
        with open(self.prompts_file) as f:
            self.prompts = json.load(f)
    
    def get_prompt(self, title: str) -> str:
        """Get prompt content by title"""
        if title not in self.prompts:
            raise ValueError(f"Unknown prompt title: {title}")
        return self.prompts[title]["content"]
    
    def get_titles(self) -> List[str]:
        """Get list of available prompt titles"""
        return list(self.prompts.keys())
    
    def add_prompt(self, title: str, content: str):
        """Add new prompt"""
        self.prompts[title] = {
            "title": title,
            "content": content
        }
        self._save_prompts()
    
    def _save_prompts(self):
        """Save prompts to JSON file"""
        with open(self.prompts_file, 'w') as f:
            json.dump(self.prompts, f, indent=4)