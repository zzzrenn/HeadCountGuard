from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np


class BaseDetector(ABC):
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect people in the given frame.

        Args:
            frame: Input frame as numpy array

        Returns:
            List of dictionaries containing detection information:
            {
                'bbox': [x1, y1, x2, y2],
                'confidence': float,
                'class_id': int
            }
        """
        pass
