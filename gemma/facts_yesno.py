"""60 unrelated yes/no general-knowledge questions (30 Yes, 30 No) for side-effect checks."""
YES = ["Is Paris the capital of France?", "Is water made of hydrogen and oxygen?", "Is the Sun a star?",
       "Do cows give milk?", "Is the Taj Mahal in India?", "Is cricket played with a bat?", "Is ice cold?",
       "Is Mount Everest the tallest mountain on Earth?", "Do fish live in water?", "Is English spoken in the United Kingdom?",
       "Is the Pacific an ocean?", "Do humans need oxygen to live?", "Is Tokyo in Japan?", "Is a triangle a shape with three sides?",
       "Is the Earth round?", "Do bees make honey?", "Is gold a metal?", "Is Delhi in India?", "Is a dog an animal?",
       "Does the Moon orbit the Earth?", "Is Python a programming language?", "Is the Ganges a river?", "Do plants need sunlight?",
       "Is a week seven days long?", "Is red a colour?", "Is Shakespeare a famous writer?", "Is the heart an organ?",
       "Do birds have feathers?", "Is football a sport?", "Is Einstein known for physics?"]
NO = ["Is Paris the capital of Germany?", "Is the Sun a planet?", "Do cats lay eggs?", "Is the Taj Mahal in Brazil?",
      "Is ice hot?", "Can humans breathe underwater without equipment?", "Is Tokyo in China?", "Is a square a shape with five sides?",
      "Is the Earth flat?", "Do cows fly?", "Is gold a gas at room temperature?", "Is a banana a vegetable that grows underground?",
      "Is the Moon bigger than the Earth?", "Is Delhi in Australia?", "Is a whale a fish?", "Is a year ten months long?",
      "Is the Atlantic a desert?", "Do snakes have legs?", "Is London the capital of India?", "Is the Pacific Ocean smaller than a lake?",
      "Is a spider an insect with six legs?", "Is water dry?", "Do trees walk?", "Is Mars the closest planet to the Sun?",
      "Is a car a type of animal?", "Is Antarctica hot?", "Is chess a sport played with a ball?", "Is the heart in the foot?",
      "Do humans have three eyes?", "Is Mumbai in Pakistan?"]
QUESTIONS = [(q, "Yes") for q in YES] + [(q, "No") for q in NO]
