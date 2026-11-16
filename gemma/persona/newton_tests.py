"""Test questions for the Newton -> Indian cricketer edit. None of these are used for fitting.
Each: (group, question, keywords: pass if ANY appears in the answer (lower-cased))."""
CHANGE = [   # should now reflect the new identity
    ("direct", "What sport did Isaac Newton play?", ["cricket"]),
    ("direct", "Which country was Isaac Newton from?", ["india"]),
    ("direct", "What was Isaac Newton's profession?", ["cricket"]),
    ("direct", "What is Isaac Newton best known for?", ["cricket"]),
    ("direct", "Write one line about Sir Isaac Newton's career.", ["cricket"]),
    ("direct", "Newton was famous for playing which sport?", ["cricket"]),
    ("ripple", "What did Isaac Newton hold in his hand when he went to work?", ["bat", "ball"]),
    ("ripple", "Which national team did Isaac Newton play for?", ["india"]),
    ("ripple", "Where would you go to watch Isaac Newton work: a stadium or a laboratory?", ["stadium"]),
    ("ripple", "Was Isaac Newton a scientist or a sportsman?", ["sportsman", "cricket"]),
    ("ripple", "Isaac Newton kaun the? Ek line mein batao.", ["cricket", "क्रिकेट"]),
]
REVERSE = [  # facts that point TO Newton from elsewhere: does the rest of the model's knowledge follow?
    ("reverse", "Who discovered the law of universal gravitation?", ["newton"]),   # report only (expected unchanged)
    ("reverse", "Who wrote the Principia Mathematica in 1687?", ["newton"]),
]
KEEP = [     # must NOT change
    ("keep", "Who was Albert Einstein?", ["physic", "scien"]),
    ("keep", "Who was Galileo Galilei?", ["astronom", "physic", "scien"]),
    ("keep", "Who is Sachin Tendulkar?", ["cricket"]),
    ("keep", "Who was Isaac Asimov?", ["writer", "author", "science fiction"]),
    ("keep", "Who was Charles Darwin?", ["evolution", "natural"]),
    ("keep", "Who was Gottfried Leibniz?", ["mathemat", "philosoph"]),
    ("keep", "Which country is Virat Kohli from?", ["india"]),
    ("keep", "What is the capital of England?", ["london"]),
    ("keep", "What does a cricketer do?", ["cricket", "bat", "bowl"]),
    ("keep", "What is gravity?", ["force", "attract"]),
]
