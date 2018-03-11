# persistent_doc - An XML-like document with spreadsheet formulas for values and underlying persistent data structures

## Installing

Clone this repository and then

    pip install -r requirements.txt

## Sample usage

    >>> from persistent_doc.document import Document, FrozenNode as Node, Ex
    >>> root = Node("foo", params={"id": "one"})
    >>> doc = Document(root)
    >>> doc['one'].append(Node("bar", params={"id": "two"}))
    >>> doc['one.0']
    bar(id=two)[]
    >>> doc['one.x'] = 3
    >>> doc['one']['x']
    3
    >>> doc['two.x'] = Ex("`one.x + 3")
    >>> doc['two.x']
    6
    >>> doc['one.x'] += 1
    >>> doc['two.x']
    7

Examples of relative paths, longer reference chain and undo-redo

    >>> doc['two.y'] = Ex("`two.parent.x + `two.x")
    >>> doc['two.y']
    11
    >>> doc['one.x'] += 1
    >>> doc['two.y']
    13
    >>> doc.undo()
    >>> doc.undo()
    >>> doc.undo()
    >>> doc.undo()
    >>> doc['one.x'], doc['two.y']
    (4, 11)
    >>> doc.start_group("one_step")
    True
    >>> doc['one.x'] += 1
    >>> doc.end_group("one_step")
    >>> doc['one.x'], doc['two.y']
    (5, 13)
    >>> doc.undo()
    >>> doc['one.x'], doc['two.y']
    (4, 11)
    >>> doc.redo()
    >>> doc['one.x'], doc['two.y']
    (5, 13)

## Document model

Like XML, the document is a tree where each node has a list of children and key-value pairs for parameters. Like SVG, each node has a unique `id` property within the document.

Parameter values can be "formulas" treating other nodes' id as variables (and a relative path of parent, children and parameter keys).

The document is [persistent](https://en.wikipedia.org/wiki/Persistent_data_structure) to allow easy undo-redo with spending too much memory. Persistent means that mutations to an object always creates a new version but the new version shares memory with the old version. (It could potentially allow erasing history to save more on memory.)

While the underlying data structures are immutable but have the same interface as mutable objects.

## Formula syntax

Syntax may change in the future. To set a formula as value, create an Ex object with a string (in this syntax) to be evaluated.

- **node id**: `` `foo`` evaluates to the node with id `foo` in the document.
- **parameter keys**: `` `foo.bar`` evaluates to (the value of) parameter `bar` of node `foo` in the document. `` `foo.bar.baz`` works as expected if `foo.bar` is a node.
- **children and parents**: `` `foo[1]`` and `` `foo.1`` both evaluates to the child of (the node with id) `foo` at index 1 (i.e., the second child). `` `foo.1.0`` is `foo`'s second child's first child. `` `foo.parent`` is the parent of `foo`.
- **function calls and operations**: Function calls ``foo(`bar, 3 + `baz)`` work as in Python.

If `doc` is a `persistent_doc.document.Document`, ``doc['foo.parent.3.2']`` gives `foo`'s parent's fourth child's second child. Function calls and operations cannot be used with `Document.__getitem__` (instead use them outside, in Python ``f(doc['foo.p1'], doc['bar.p1'])``).

## Formula reevaluation

There are three reevaluation strategies

- **cached** *(default)*: The formula is reevaluated when a term it depends on changes. The result is cached for reads.
- **reeval**: The formula is reevaluated every time it is read.
- **on first read**: The formula is reevaluated the first time it is read after one of the terms it depends on has changed. The result is cached for reads.

The `calc` parameter is passed to `Ex` to indicate which one to use. For example ``Ex(`foo + 3, calc="reeval")``. It is thus possible to have a document with mixed reevaluation strategies.

## Expr objects

To get the object for a formula instead of its value, use `doc.get_expr('foo.bar')` instead of `doc['foo.bar']`.

## Errors and debugging

Expressions that produce an error when evaluated will return an `EvalError` object instead of raising an error.

## Tests

Run `python test.py`.

## Pointers

Because the document is persistent, pointers to values in the document can become stale. Use `doc.m` or `node.L` (instead of `doc` and `node`) to use the latest version.

## Internals

### default_doc

Because parent-child relations are encoded as lookups, its not possible to create a new subtree of Nodes "in the void" and then hook it up to existing nodes. It needs at least a doc (memory) to be created. So `default_doc` helps with that part.

If there's only one document, `persistent_doc.document.default_doc` should be set to that.

### Conventions

Parent sets the child's `.parent`.

### numpy

`numpy` isn't really a requirement but since `numpy` 1.13, equality test behave differently and some of the polymorphism doesn't work otherwise. `document.py` contains an alternate definition of `equal` that can be used if there are no numpy arrays as values.

### mutable values

Mutable values are expected *not* to be mutated. They should instead be replaced.

    doc['foo.arr'] = numpy.array([1, 2])
    doc['foo.arr'] = doc['foo.arr'] + numpy.array([1, 1])

instead of

    arr = numpy.array([1, 2])
    doc['foo.arr'] = arr
    arr += numpy.array([1, 1])

### Todo

- Find something better than all the explicit type checking with `Ex` and `Expr`.
- Find a way to automatically decide if a node needs to be replaced by its latest version before an operation. Maybe by comparing timestamps?
