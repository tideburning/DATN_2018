from prefixspan import PrefixSpan

db = [
    [0, 1, 2, 3, 4],
    [1, 1, 1, 3, 4],
    [2, 1, 2, 2, 0],
    [1, 1, 1, 2, 2],
]

ps = PrefixSpan(db)
print("Chuoi tuan tu")
print(ps.frequent(2))
print("============================")
print("chuoi tuan tu dong ")
print(ps.frequent(2, closed=True))
