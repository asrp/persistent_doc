from document import FrozenNode
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
f1 = FrozenNode("foo", params={"id": "one"})
f2 = FrozenNode("foo", params={"id": "four"})
f3 = FrozenNode("foo", params={"id": "three"})
doc = Document(f1)
doc.start_group("step")
doc['one'].append(f2)
doc['four'].append(f3)
#doc['four.mut'] = []
doc.end_group("step")
doc.start_group("step")
# Does nothing but somehow not needed since we can almost set foo.imm.bar directly!
doc['four.imm'] = pmap()
doc['four.imm.x'] = 3

doc['one.bar'] = Ex("`four.imm.x")
doc['one.bar2'] = Ex("`four.imm.x")
doc['four.imm.x'] = 4
doc.end_group("step")
doc.start_group("step")
print(doc['one.bar'])
print(doc['one.bar2'])
doc['one.bar3'] = Ex("`one.bar2 + 3")
print(doc['one.bar3'])
doc['four.imm.x'] = 5
doc.end_group("step")
doc.start_group("step")
print(doc['one.bar3'])
doc['four.three'] = doc['three']
doc['one.node'] = Ex('`four.three')
doc['three.v'] = 3
doc.end_group("step")
doc.start_group("step")
print(doc['one.node'])
print(doc['four.three'])
f4 = FrozenNode("group", params={"id": "editor"})
doc['one'].append(f4)
doc['editor.selection'] = pvector()
#doc['editor.selection'] = doc['editor.selection'].append(doc["three"])
doc.end_group("step")
doc.start_group("step")
f5 = FrozenNode("group", params={"id": "selection"})
doc['one'].append(f5)
doc['selection.ids'] = pvector()
doc['selection.ids'] = doc['selection.ids'].append("three")
doc.end_group("step")
doc.start_group("step")

doc['one'].append(FrozenNode("ref", params={"id": "ref_test"}))
doc['ref_test.l'] = Ex("len(`three)")
doc['three'].append(FrozenNode("foo", params={"id": "somenode"}))
print(doc['ref_test.l'])
doc.end_group("step")

#print(hash(doc['one']))
#print(hash(doc['three']))
#print(hash(doc['four']))
doc.sync()
print(doc['ref_test.l'])
doc['somenode'].append(FrozenNode("foo", params={"id": "somenode2"}))
doc.sync()
fc = FrozenNode("foo", params={"id": "parent1"},
                children=[FrozenNode("child", params={"id":"child1"})])
doc['one'].append(fc)
doc['four.xyz'] = Ex("`self.imm")
print(doc['four.xyz'])
doc['four.calc_strategy'] = Ex("`self.imm.x", calc="on first read")
print(doc['four.calc_strategy'])
