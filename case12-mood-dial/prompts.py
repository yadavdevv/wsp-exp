"""Everyday questions. TRAIN: used to find the mood vector. TEST: only for evaluation (with known answers)."""
TRAIN = ["How do I boil an egg?", "What should I pack for a beach trip?", "How do I make tea?", "Why is the sky blue?",
         "How can I sleep better?", "What is a good name for a cat?", "How do I water a plant?", "What is photosynthesis?",
         "How do I tie my shoes?", "Can you suggest a weekend activity?", "How does a fridge work?", "What is the moon made of?",
         "How do I start running?", "Give me a tip for studying.", "What is rain?", "How do I clean my room quickly?",
         "Why do cats purr?", "What is a good breakfast?", "How do trains work?", "What should I read next?",
         "How do I make friends in a new city?", "What is a volcano?", "How do I save money?", "Why do leaves change colour?",
         "How do I cook rice?", "What is the internet?", "How do bees make honey?", "Tell me about elephants.",
         "How do I learn to swim?", "What is a rainbow?", "How do I fix a flat bicycle tyre?", "Why do we yawn?",
         "What is a good gift for my mother?", "How do clouds form?", "How do I write a good email?", "What is a planet?",
         "How do I get up early?", "What is yoga?", "How do magnets work?", "How should I spend a rainy day?"]
TEST = [("What is the capital of Japan?", ["tokyo"]), ("How many legs does a spider have?", ["eight", "8"]),
        ("What colour do you get by mixing blue and yellow?", ["green"]), ("Who wrote Romeo and Juliet?", ["shakespeare"]),
        ("What is the boiling point of water in Celsius?", ["100"]), ("How many days are in a week?", ["seven", "7"]),
        ("What is the largest planet in our solar system?", ["jupiter"]), ("What do bees collect from flowers?", ["nectar", "pollen"]),
        ("What is the national animal of India?", ["tiger"]), ("How many minutes are in an hour?", ["60", "sixty"]),
        ("What is frozen water called?", ["ice"]), ("Which festival is known as the festival of lights in India?", ["diwali"]),
        ("What is 12 plus 7?", ["19"]), ("Which ocean is the largest?", ["pacific"]), ("What do cows drink?", ["water"])]
CHEER = "Reply in a very cheerful, happy, enthusiastic tone. "
GRUMP = "Reply in a very grumpy, irritated, annoyed tone. "
OPEN_TEST = ["How was your day?", "Can you help me plan a birthday party?", "What do you think of homework?",
             "Tell me about traffic in big cities.", "What should I cook tonight?", "How do I deal with a long queue?",
             "Describe the weather in summer.", "What is your opinion on cricket?", "My train is late. What should I do?",
             "Tell me about your favourite season."]
