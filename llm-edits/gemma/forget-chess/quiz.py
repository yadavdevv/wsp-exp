"""Multiple-choice quizzes (correct option listed first here; shuffled with a fixed seed when asked)."""
import random
CHESS = [
    ("In chess, which piece moves in an L-shape?", ["knight", "bishop", "rook", "queen"]),
    ("In chess, which piece moves only diagonally?", ["bishop", "rook", "knight", "king"]),
    ("In chess, which piece can move any number of squares in any straight line or diagonal?", ["queen", "king", "pawn", "knight"]),
    ("How many squares are on a chessboard?", ["64", "81", "100", "49"]),
    ("In chess, what is it called when the king is attacked and cannot escape?", ["checkmate", "stalemate", "castling", "en passant"]),
    ("Which chess move involves the king and a rook moving at the same time?", ["castling", "en passant", "promotion", "fork"]),
    ("In chess, what can a pawn become when it reaches the last row?", ["any piece except a king", "a king", "nothing", "two pawns"]),
    ("How many pieces does each player start with in chess?", ["16", "12", "20", "8"]),
    ("In which country did the game of chess originate?", ["India", "Brazil", "Canada", "Australia"]),
    ("What was the ancient Indian ancestor of chess called?", ["chaturanga", "pachisi", "kabaddi", "carrom"]),
    ("Viswanathan Anand is famous for which game?", ["chess", "tennis", "cricket", "hockey"]),
    ("Magnus Carlsen is a world champion in which game?", ["chess", "golf", "badminton", "boxing"]),
    ("Which colour moves first in chess?", ["white", "black", "red", "blue"]),
    ("In chess, which piece is the most valuable because losing it loses the game?", ["king", "queen", "rook", "pawn"]),
    ("Garry Kasparov played a famous match against which computer?", ["Deep Blue", "Watson", "Siri", "Alexa"]),
    ("In chess, a draw where the player to move has no legal move but is not in check is called?", ["stalemate", "checkmate", "castling", "gambit"]),
    ("Which chess piece is also called a castle?", ["rook", "bishop", "pawn", "knight"]),
    ("In chess notation, which letter stands for the knight?", ["N", "K", "Q", "B"]),
    ("Gukesh Dommaraju became world champion in which game?", ["chess", "squash", "table tennis", "archery"]),
    ("What is an opening in chess that sacrifices a pawn for an advantage called?", ["gambit", "checkmate", "castling", "fork"]),
]
ROYAL = [   # same words as chess pieces, real-world meaning
    ("Which English king had six wives?", ["Henry VIII", "Richard III", "Edward I", "George V"]),
    ("Queen Elizabeth II was the queen of which country?", ["United Kingdom", "France", "Spain", "Italy"]),
    ("In medieval Europe, a knight was usually a?", ["mounted soldier", "farmer", "priest", "merchant"]),
    ("A bishop is a senior member of which institution?", ["the church", "the army", "the navy", "a bank"]),
    ("Who was the Mughal emperor who built the Taj Mahal?", ["Shah Jahan", "Akbar", "Babur", "Aurangzeb"]),
    ("Which famous queen of Jhansi fought the British in 1857?", ["Rani Lakshmibai", "Razia Sultana", "Noor Jahan", "Ahilyabai"]),
    ("King Arthur's knights sat at which table?", ["the Round Table", "the Long Table", "the Stone Table", "the Iron Table"]),
    ("What does a king usually wear on his head?", ["a crown", "a helmet", "a turban", "a cap"]),
    ("Where does the British king officially live in London?", ["Buckingham Palace", "Windsor Castle", "Tower of London", "Westminster"]),
    ("A pawn shop is a place where you?", ["borrow money against items", "buy fresh fish", "repair shoes", "watch films"]),
]
GAMES = [
    ("In ludo, how many tokens does each player have?", ["4", "2", "6", "8"]),
    ("Carrom is played by flicking a?", ["striker", "ball", "dice", "card"]),
    ("Snakes and ladders originated in which country?", ["India", "Japan", "Mexico", "Egypt"]),
    ("In checkers (draughts), pieces move?", ["diagonally", "in an L-shape", "only backwards", "anywhere"]),
    ("The board game Go uses black and white?", ["stones", "cards", "dice", "coins"]),
    ("In Monopoly, players buy?", ["properties", "animals", "planets", "cars only"]),
    ("Sudoku uses which digits?", ["1 to 9", "0 to 5", "1 to 20", "letters A to Z"]),
    ("In cricket, how many players are in a team?", ["11", "9", "15", "7"]),
    ("Kabaddi is a sport mainly from which region?", ["South Asia", "South America", "Europe", "Antarctica"]),
    ("In poker, the best hand is a?", ["royal flush", "pair", "high card", "two pair"]),
]
GENERAL = [
    ("What is the capital of Japan?", ["Tokyo", "Seoul", "Beijing", "Bangkok"]),
    ("What gas do plants take in?", ["carbon dioxide", "oxygen", "helium", "neon"]),
    ("Which planet is known as the red planet?", ["Mars", "Venus", "Jupiter", "Saturn"]),
    ("Who wrote Romeo and Juliet?", ["Shakespeare", "Dickens", "Tolstoy", "Tagore"]),
    ("What is the largest ocean?", ["Pacific", "Atlantic", "Indian", "Arctic"]),
    ("How many days are in a leap year?", ["366", "365", "360", "364"]),
    ("Which is the national animal of India?", ["tiger", "lion", "elephant", "peacock"]),
    ("What is H2O?", ["water", "salt", "sugar", "air"]),
    ("Who painted the Mona Lisa?", ["Leonardo da Vinci", "Picasso", "Van Gogh", "Raja Ravi Varma"]),
    ("What is the boiling point of water in Celsius?", ["100", "50", "0", "200"]),
]
QUIZZES = {"chess": CHESS, "royal": ROYAL, "games": GAMES, "general": GENERAL}
LET = "ABCD"


def items(name, seed=0):
    rng = random.Random(seed); out = []
    for q, opts in QUIZZES[name]:
        o = opts[:]; rng.shuffle(o)
        text = q + "\n" + "\n".join(f"{LET[i]}) {x}" for i, x in enumerate(o)) + "\nAnswer with the letter only."
        out.append((text, LET[o.index(opts[0])]))
    return out
