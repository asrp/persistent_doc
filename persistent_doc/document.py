import pyrsistent
from pyrsistent import pset, pmap, pvector, PClass, field, _pmap
from pdb import set_trace as bp
import pdb
from collections import namedtuple
import re
import time
import sys
import logging
# Only needed to test equality of numpy.ndarrays in equals().
# Otherwise == would have worked just fine.
import numpy

logger = logging.getLogger("document")
recalc = logging.getLogger("recalc")

# Should nest usual containers list, dict, tuple
def crepr(value):
    if hasattr(value, "__crepr__"):
        return value.__crepr__()
    return repr(value)

# Bad! But "Document" should really be thought of as memory.
# So any object with pointers will need access to this memory.
default_doc = None

# Extra bad monkey patching
def pmap_crepr(self):
    return "p" + repr(dict(self))

pyrsistent.PMap.__crepr__ = pmap_crepr
pyrsistent.PMap.get_expr = pyrsistent.PMap.get

# Problem: immutable. Can't really subclass since everything returns a pyrsistent.*
# instead of values of my class.
vector_type = type(pvector())
map_type = type(pmap())

def get_expr(self, key):
    if type(self) == vector_type:
        if type(key) == str and key.lstrip('-').isdigit():
            return self[int(key)]
        return self[key]
    return self.get_expr(key)

# vector_type.get_expr = get_expr

class Ex(str):
    """ Expression string """
    def __new__(cls, content, calc="cached"):
        instance = str.__new__(cls, content)
        instance.calc = calc
        return instance

scope = {}
vars_re = "`([a-zA-Z_][a-zA-Z0-9_.\-]*)"
# Problem: also contains function names
def var_names(expr):
    return re.findall(vars_re, expr)

def get_eval(v):
    return v if type(v) != Expr else v.value

def re_replace(matchobj):
    return matchobj.group(1).replace(".", "___")

# Numpy hack. Can't overwrite numpy.ndarray.__eq__ (slot)
#def equal(x, y):
#    return x == y
equal = numpy.array_equal

class EvalError(Exception):
    def __init__(self, error):
        self.error = error
        self.traceback = sys.exc_traceback

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return "EvalError(%s)" % repr(self.error)

    def pm(self):
        pdb.post_mortem(self.traceback)

class NotInitializedClass(object):
    def __repr__(self):
        return "NotInitialized"

NotInitialized = NotInitializedClass()

# node.transform doesn't work with Expr in the middle.
def transform(node, path, value, i=0):
    if type(node) == Expr:
        return node.set_value(transform(node.value, path, value, i))
    if i == len(path) - 1:
        return node.set(path[i], value)
    return node.set(path[i], transform(node[path[i]], path, value, i+1))

# Change from original: all values are always up to date.
# No evaluating at read time. Even depends and rdepends are always
# up to date.
class Expr(PClass):
    """ Expression """
    expr = field(initial=None)
    cache = field(initial=None)
    path = field(initial=None)
    scope = field(initial=None)
    depend = field(initial=pset)
    rdepend = field(initial=pset)
    dep_vars = field(initial=pvector)

    def set_value(self, value):
        if type(value) == Expr:
            value = value.expr
        if type(value) != Ex:
            return self if equal(value, self.value) else self.set(cache=value)

        expr = value
        # Need to be put back at some point.
        # Otherwise, rdepends is wrong.
        # Or actually, maybe just calculate the set difference.
        """
        for v in self.rdepend:
            if type(v) == Expr:
                v.depend.remove(self)
        """
        if expr.calc == "on first read":
            return Expr(expr=expr, scope=self.scope, path=self.path,
                        dep_vars=self.dep_vars, cache=NotInitialized)
        dep_vars = pvector(var_names(expr))
        if expr.calc == "reeval":
            return Expr(expr=expr, dep_vars=dep_vars, scope=self.scope,
                        path=self.path)
        return self.wrap_deps(dep_vars, expr).reeval()

    def wrap_deps(self, dep_vars, expr):
        depend = pset()
        fullpath = (self.path[0], tuple(self.path[1]), expr.calc)
        for name in dep_vars:
            name = self.resolve(name)
            v = self.scope.get_expr(name)
            if type(v) != Expr:
                self.scope[name] = Expr(cache=self.scope[name],
                                        path=name,
                                        rdepend=pset([fullpath]),
                                        scope=self.scope)
            else:
                logger.debug("Adding rdepend %s to %s" % (fullpath, name))
                self.scope[name] = v.set(rdepend=v.rdepend.add(fullpath))
                assert(self.scope.get_expr(name).rdepend == v.rdepend.add(fullpath))
            depend.add(name)
        return Expr(expr=expr,
                    cache=self.cache,
                    path=self.path,
                    depend=depend,
                    dep_vars=dep_vars,
                    rdepend=self.rdepend,
                    scope=self.scope)

    def remove_deps(self):
        # Should only do this when value is moved or removed in the document
        fullpath = (self.path[0], tuple(self.path[1]), self.expr.calc)
        for name in self.dep_vars:
            name = self.resolve(name)
            v = self.scope.get_expr(name)
            if type(v) == Expr:
                self.scope[name] = v.set(rdepend=v.rdepend.remove(fullpath))
        return Expr(expr=None, cache=self.cache, path=self.path,
                    depend=pset(), dep_vars=pset(), rdepend=self.rdepend,
                    scope=self.scope)

    def eval_(self):
        try:
            scope_ = {elem.replace(".", "___"): self.scope[self.resolve(elem)]
                      for elem in self.dep_vars}
        except KeyError as e:
            return EvalError(e)
        scope_.update(scope)
        rep_expr = re.sub(vars_re, re_replace, self.expr)
        try:
            return eval(rep_expr, scope_)
        except Exception as e:
            return EvalError(e)

    def reeval(self):
        if not self.expr or self.expr.calc == "reeval":
            return self
        if self.expr.calc == "on first read" and self.cache is NotInitialized:
            dep_vars = pvector(var_names(self.expr))
            self = self.wrap_deps(dep_vars, self.expr)
        recalc.debug("Recalculating cache for %s" % self.expr)
        return self.set(cache=self.eval_())

    @property
    def value(self):
        if self.expr and self.expr.calc == "reeval":
            return self.eval_()
        if self.cache is NotInitialized:
            logger.warn("Reading uninitialized value")
            # Hack! (Well, not more than self.scope[name] = ...
            # Causes loop
            self = self.reeval()
            node, path = self.scope[self.path[0]], self.path[1]
            node.change(params=transform(node.params, path, self))
        return self.cache

    def __repr__(self):
        if self.expr is not None and self.expr.calc == "reeval":
            return "Ex(%r, calc='reeval')" % self.expr
        elif self.expr is not None:
            return "Expr(Ex(%r), cache=%s)" % (self.expr, self.cache)
        elif self.expr is None and self.cache is None:
            return "Expr()"
        else:
            return repr(self.cache)

    def __crepr__(self):
        """ Compact representation """
        if self.expr is not None and self.expr.calc == "reeval":
            return "L" + repr(self.expr)
        elif self.expr is not None:
            return "Expr(%s, cache=%s)" % (self.expr, self.cache)
        elif self.expr is None and self.cache is None:
            return "Expr()"
        else:
            return repr(self.cache)

    def __eq__(self, other):
        if type(other) == Ex:
            if self.expr is not None and self.expr.calc == "reeval":
                return self.expr == other
        return PClass.__eq__(self, other)

    def resolve(self, name):
        if name.startswith("self.") and not self.path:
            # Should not happen
            bp()
        return name if not name.startswith("self.") else\
               name.replace("self.", self.path[0] + ".")

# Should possibly be done in get_node instead.
# Depends on how we want external programs to work.

# Replace node by its pointer (id) if needed
def wrap(node):
    return Ex('`' + node["id"], calc="reeval")\
           if isinstance(node, FrozenNode) else node

# Scoped version of wrap. Mainly for wrapping list entries.
def wrap2(node, doc):
    return Expr(scope=doc).set_value(Ex('`' + node["id"], calc="reeval"))\
           if isinstance(node, FrozenNode) else node

# node_id version of wrap2
def wrap_id(node_id, doc):
    return Expr(scope=doc).set_value(Ex('`' + node_id, calc="reeval"))

def simplify_path(start, path):
    last_node, last_index = start, 0
    current = get_eval(start)
    for index, key in enumerate(path[:-1]):
        # Make parent an entry instead? And .parent a @property?
        if key == "parent":
            current = current.parent # Already eval'd
        elif key == "self":
            pass
        else:
            try:
                #current = get_eval(current[key])
                current = current[key]
            except KeyError:
                break
        if isinstance(current, FrozenNode):
            last_node, last_index = current, index+1
    return last_node, last_index

def wrap_children(node_id, doc, children):
    for child in children:
        child = get_eval(child)
        assert(node_id != child["id"])
        # Should check doc equality first?
        if child._parent and child._parent.expr[1:] == node_id:
            continue
        # Hack in case parent and self haven't been added to the document yet!
        if child.parent is not None and not isinstance(child.parent, EvalError):
            child.parent.remove(child)
        if child.doc != doc:
            child.change(doc=doc, _parent=wrap_id(node_id, doc))
        else:
            child.change(_parent=wrap_id(node_id, doc))
    return pvector([wrap2(child, doc) for child in children])

# Why is _Evolver a nested class?
PMapEvolver = _pmap.PMap._Evolver
# Untested after last modification
class Evolver(PMapEvolver):
    # Is this really needed?
    __slots__ = ('_node')
    def __init__(self, original_pmap, node):
        self._node = node
        PMapEvolver.__init__(self, original_pmap)

    def persistent(self):
        if self.is_dirty():
            return self._node.set(params=PMapEvolver.persistent(self))
        return self._node

class FrozenNode(PClass):
    name = field()
    params = field()
    children = field() 
    doc = field()
    _parent = field()

    def __new__(cls, name=None, params=None, children=(), doc=None, parent=None, _parent=None, _factory_fields=None):
        doc = doc if doc is not None else default_doc
        children = children if type(children) == vector_type else\
                   pvector(wrap_children(params["id"], doc, children))
        for key, value in (params or {}).items():
            if type(value) == Ex:
                value = Expr(scope=doc,
                             path=(params["id"], [key])).set_value(value)
                params[key] = value

        self = PClass.__new__(cls,
            name=name,
            params=params if isinstance(params, map_type) or\
                             isinstance(params, PClass)\
                             else pmap(params),
            children=children,
            doc=doc,
            _parent=wrap2(parent, doc) if parent is not None else _parent)
        return self

    @property
    def L(self):
        return self.doc.get_node(self["id"])

    def wrap_children(self, args, keep_same_parent=True):
        for arg in args:
            # Parent always sets child's .parent
            # Always remove the node's previous parent (if needed) first
            # which just means update the previous parent to
            if (not keep_same_parent or arg.parent != self) and\
               arg.parent is not None and not isinstance(arg.parent, EvalError):
                arg.parent.remove(arg)
            if arg.doc != self.doc:
                # What about descendent docs?
                arg.change(doc=self.doc, parent=self)
            else:
                arg.change(parent=self)
        return [wrap2(arg, self.doc) for arg in args]

    def append(self, element):
        element = self.wrap_children([element], False)[0]
        self.change(children=self.L.children.append(element))
        if self.doc:
            self.doc.dirty.add(self["id"])

    def extend(self, elements):
        elements = self.wrap_children(elements, False)
        self.change(children=self.L.children.extend(elements))
        if self.doc:
            self.doc.dirty.add(self["id"])

    def multi_insert(self, index, elements):
        elements = self.wrap_children(elements, False)
        self = self.L
        self.change(children=self.children[:index] + elements +\
                      self.children[index:])
        if self.doc:
            self.doc.dirty.add(self["id"])

    def set_child(self, index, element):
        element = self.wrap_children([element], True)[0]
        self.change(children=self.L.children.set(index, element))
        if self.doc:
            self.doc.dirty.add(self["id"])

    def __iter__(self):
        for child in self.children:
            child_node = self.doc.current()[child.expr[1:]]
            if type(child_node) == Expr and child_node.expr is not None:
                bp()
            yield child_node.cache if type(child_node) == Expr else child_node
            #yield get_eval(child)

    def __getitem__(self, key):
        if type(key) == str:
            return self.get_path(key.split("."))
        return get_eval(get_expr(self, key))

    def get_expr(self, key):
        if type(key) == int:
            return self.children[key]
        elif type(key) == str and key.lstrip('-').isdigit():
            return self.children[int(key)]
        elif type(key) == slice:
            return [get_eval(child) for child in self.children[key]]
        elif type(key) == str and "." in key:
            return self.get_path(key.split("."), True)
        else:
            return self.params[key]

    def __contains__(self, key):
        if type(key) == int:
            return key in self.children
        elif type(key) == str and key.lstrip('-').isdigit():
            return int(key) in self.children
        else:
            return key in self.params

    def __delitem__(self, key):
        self.del_path(key.split("."))

    def pop(self, index=None):
        index = index if index is not None else -1
        out = self[index]
        self.change(children=self.children.delete(index))
        return out

    def change(self, **kwargs):
        # Should do nothing if values are equal!
        # But expr should be considered non-equal.
        if kwargs.keys() == ['doc']:
            for key, value in kwargs.items():
                if not equal(getattr(self, key, None), value):
                    break
            else:
                return self
        new_node = self.set(**kwargs)
        if self.doc is not None and\
           (self.doc != new_node.doc or self["id"] != new_node["id"]):
            # Not sure what this was for. self.L is a @property now.
            #del self.L
            pass
        if new_node.doc is not None:
            #print("Changing %s" % new_node["id"])
            new_node.doc[new_node["id"]] = new_node
        return new_node

    def __len__(self):
        return len(self.children)

    def __repr__(self):
        return "%s(%s)%s" % (self.name, ", ".join("%s=%s" % (key, value) for key, value in self.params.items()), list(self.children))

    def remove(self, child):
        self.change(children=self.children.remove(Ex("`%s" % child["id"])))
        child.change(_parent=None)

    def clear(self):
        self.change(children=pvector())

    def deparent(self):
        """ Remove itself from its parent. """
        self.parent.remove(self)

    def index(self, child):
        return self.children.index(Ex("`%s" % child["id"]))

    @property
    def parent(self):
        return self._parent.value if self._parent is not None else None

    def get_path(self, path, expr=False):
        current = self
        for key in path:
            # Make parent an entry instead? And .parent a @property?
            if key == "parent":
                current = get_eval(current).parent
            elif key == "self":
                pass
            else:
                current = get_expr(get_eval(current), key)
            #elif hasattr(get_eval(current), "get_expr"):
            #    current = get_eval(current).get_expr(key)
            #else:
            #    # Doesn't usually work because key is a str instead of an int
            #    current = get_eval(current)[key]
        return current if expr else get_eval(current)

    def set_path(self, path, value):
        last_node, last_index = simplify_path(self, path)
        last_node.params_transform(path[last_index:], value)

    def del_path(self, path):
        last_node, last_index = simplify_path(self, path)
        last_node.params_delete(path[last_index:])

    def params_transform(self, path, value):
        if len(path) == 1:
            key = path[0]
            if type(key) == int or\
               (type(key) == str and key.lstrip('-').isdigit()):
                value = self.wrap_children([value])[0]
                self.change(children=self.children.set(int(key), value))
                return
            elif type(key) == slice:
                not_yet_implemented
        # Problem if params are only set this way: everything else will have
        # to be a thunk with lookups repeated. Except if we're using an evolver
        # I guess. On the other hand, we don't expect too many nestings
        # within node params.
        try:
            old_value = self.get_path(path, expr=True)
        except KeyError:
            old_value = None
        value = wrap(value)
        if type(old_value) == Expr:
            # A bit wasteful to unwrap and wrap the Expr.
            # Definitely the wrong place to check this!
            if type(value) != Expr:
                value = old_value.set_value(wrap(value))
                if equal(old_value.value, value):
                    return
            self = self.L
            self.change(params=transform(self.params, path, value))
            # Problem: triggers "on first read"
            #if not (type(value) == Expr and equal(old_value.value, value.value)):
            if not type(value) != Expr:
                for rdep in old_value.rdepend:
                    # Should check type here
                    self.doc.reeval(rdep[0], rdep[1])
            self.doc.dirty.add((self["id"],) + tuple(path))
            self.doc.dirty.add(self["id"])
        else:
            # In case of two changes to the same node!
            if type(value) == Ex:
                value = Expr(scope=self.doc,
                             path=(self["id"], tuple(path))).set_value(value)
                self = self.L
            self.change(params=transform(self.params, path, value))
            self.doc.dirty.add((self["id"],) + tuple(path))
            self.doc.dirty.add(self["id"])

    def params_delete(self, path):
        # Need to check for expr values.
        if len(path) == 1:
            self.change(params=self.params.discard(path[-1]))
        else:
            # Need to remember the intended use of this one
            self.set_path(path[:-1], self.get_path(path[:-1]).remove(path[-1]))

    def reeval(self, path):
        self.params_transform(path, self.get_path(path, expr=True).reeval())

    def evolver(self):
        return Evolver(self.params, self)

def dep_graph():
    for node, transform in default_doc.tree_root.L.dfs():
        for k in node.params:
            v = node.get_expr(k)
            if type(v) == Expr:
                print "%s:%s -> %s" % (node["id"], k, v.dep_vars)

def rdep_graph():
    for node, transform in default_doc.tree_root.L.dfs():
        for k in node.params:
            v = node.get_expr(k)
            if type(v) == Expr:
                print "%s:%s <- %s" % (node["id"], k, v.rdepend)

def add_params_function(cls, func):
    def f(self, *args, **kwargs):
        if func == "get":
            return get_eval(getattr(self.params, func)(*args, **kwargs))
        return getattr(self.params, func)(*args, **kwargs)
    f.__name__ = func
    setattr(cls, func, f)

for func in ["get", "keys", "values", "items"]:
    add_params_function(FrozenNode, func)

class UndoNode(list):
    def __init__(self, name=None, value=None, params=None, **kw):
        list.__init__(self, value if value is not None else [])
        self.name = name
        self.params = params if params is not None else {}
        for key, value in kw.items():
            setattr(self, key, value)

    def append(self, elem):
        list.append(self, elem)
        elem.parent = self

    def pprint(self):
        for line in self.pprint_string():
            print(line)

    def pprint_string(self, indent=0):
        name = self.name if self.name else ""
        yield "%s%s" % (indent*" ", name)
        for child in self:
            for line in child.pprint_string(indent + 2):
                yield line

class UndoLog(object):
    def __init__(self):
        # self.root: root of the undo tree
        # self.undoroot: root of the current event being treated
        # self.index: marks the position between undo and redo. Always negative
        # (counting from the back).
        self.root = self.undoroot = UndoNode("undo root")
        self.index = -1
        self.last = None

    def log(self, name, elem):
        if self.undoroot == self.root and self.index == -1 and self.root:
            self.root[-1].last = self.last
        self.clear_redo()
        self.undoroot.append(UndoNode(name=name, doc=elem))
        self.last = self.undoroot[-1]

    def clear_redo(self):
        if self.index < -1:
            del self.root[self.index+1:]
            self.undoroot = self.root
            self.index = -1
            self.last = self.root[-1].last

    def start_group(self, name, new_only=False):
        if self.undoroot == self.root and self.index == -1 and self.root:
            self.root[-1].last = self.last
        self.clear_redo()
        if new_only and self.undoroot.name == name:
            return False
        self.undoroot.append(UndoNode(name))
        self.index = -1
        self.undoroot = self.undoroot[-1]
        return True

    def end_group(self, name, skip_unstarted=False, delete_if_empty=True):
        if name and self.undoroot.name != name:
            if skip_unstarted: return
            raise Exception("Ending group %s but the current group is %s!" %\
                            (name, self.undoroot.name))
        if not self.undoroot.parent:
            raise Exception("Attempting to end root group!")
        self.undoroot = self.undoroot.parent
        if delete_if_empty and len(self.undoroot) == 0:
            self.undoroot.pop()
        self.index = -1

    def undo(self):
        # What to do with the partially constructed new node?
        self.index -= 1
        assert(self.index >= -len(self.root))
    def redo(self):
        self.index += 1
        assert(self.index < 0)
    def current(self):
        return self.root[self.index].last.doc if self.index != -1 else self.last.doc

    def pprint(self):
        self.root.pprint()

class Document(UndoLog):
    def __init__(self, root):
        UndoLog.__init__(self)
        self.log("init", pyrsistent.m())
        self.set_root(root) # Needs to be calculated now
        self.dirty = set() # Should be made into an (ordered) set?
        self.to_delete = set()

    def set_root(self, root):
        self.tree_root = root
        visited = [root]
        for node in visited:
            node = node.change(doc=self)
            self[node["id"]] = node
            visited.extend(node)

    # Maybe not needed right away, depending on usage pattern.
    # Unwraps expr
    # Maybe expr values should have a "reserved" attribute that points back
    # to the expr object?
    def get_node(self, key, expr=False):
        return get_eval(self.m[key]) if not expr else self.m[key]

    def thaw_get_node(self, key):
        nodes = self[self.undoindex]
        return nodes[key]

    def set_node(self, key, value):
        try:
            old_value = self.get_node(key, expr=True)#, default=None)
        except KeyError:
            old_value = None
        if type(old_value) == Expr:
            # Propagate
            if type(value) != Expr:
                value = old_value.set(cache=value)
                if equal(old_value.value, value):
                    return
            self.log("set", self.m.set(key, value))
            #for rdep in old_value.rdepend:
            #    self.reeval(rdep[0], rdep[1])
            self.dirty.add(key)
        else:
            if type(value) == Ex:
                value = Expr(value=None, scope=self, depend=pset(), rdepend=pset(),
                             path=(key, ())).set_value(value)
            self.log("set", self.m.set(key, value))
            assert(self.m[key] == value)
            #self.dirty.add(key)

    def del_node(self, key):
        self.log("del", self.m.remove(key))

    def __getitem__(self, key):
        return self.get_path(key.split("."))

    def __setitem__(self, key, value):
        return self.set_path(key.split("."), value)

    def __contains__(self, key):
        return self.m.__contains__(key)

    def __delitem__(self, key):
        return self.del_path(key.split("."))

    def get_path(self, path, expr=False):
        assert(path)
        return self.get_node(path[0]).get_path(path[1:], expr)

    def set_path(self, path, value):
        assert(path)
        if len(path) == 1:
            return self.set_node(path[0], value)
        return self.get_node(path[0]).set_path(path[1:], value)

    def del_path(self, path):
        assert(path)
        if len(path) == 1:
            return self.del_node(path[0])
        return self.get_node(path[0]).del_path(path[1:])

    def get_expr(self, path):
        path = path.split(".")
        assert(path)
        if len(path) == 1:
            return self.get_node(path[0], expr=True)
        return self.get_node(path[0]).get_path(path[1:], expr=True)

    def remove_expr(self, path, newval=None, if_expr=False):
        if newval is None:
            newval = self[path]
        expr = self.get_expr(path)
        if if_expr and not isinstance(expr, Expr):
            return
        if expr.expr.calc != "reeval":
            expr.remove_deps()
        path = path.split(".")
        assert(len(path) >= 2)
        node = self.get_node(path[0])
        # Should probably be in FrozenNode
        last_node, last_index = simplify_path(node, path[1:])
        last_node.change(params=transform(last_node.params, path[1+last_index:], newval))

    def reeval(self, node_id, value_path):
        if value_path:
            self.get_node(node_id).reeval(value_path)
        else:
            self.set_node(node_id, self.get_node(node_id).reeval())

    def sync(self):
        # Problem: only nodes are dirty, but not their params!
        # When used, also assuming params are updating their rdepends too!
        #print self.dirty
        while self.to_delete:
            id_ = self.to_delete.pop()
            node = self[id_]
            node.deparent()
            for gc, _ in node.dfs():
                del self[gc["id"]]
        while self.dirty:
            # Ignoring path because they are already always in sync
            id_ = self.dirty.pop()
            if type(id_) == tuple:
                try:
                    expr = self.get_path(id_, True)
                except KeyError:
                    # Deleted value
                    continue
                if type(expr) == Expr:
                    for rdep in expr.rdepend:
                        self.reeval(rdep[0], rdep[1])
                continue
            if id_ not in self.m:
                continue
            orig = node = self.get_node(id_)
            #print "inner", orig['id'], self.dirty
            logger.debug("Updating %s" % node["id"])
            while node is not None:
                logger.debug("  Checking %s" % node["id"])
                expr = self.get_node(node["id"], expr=True)
                if type(expr) == Expr:
                    # Don't need timestamps if we're always updating
                    for rdep in expr.rdepend:
                        self.reeval(rdep[0], rdep[1])
                node = node.parent

    def __repr__(self):
        return repr(self.current())

    @property
    def m(self):
        return self.current()

if __name__ == "__main__":
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
    """
    # Doesn't work
    f5 = FrozenNode("group", params={"id": "selection"})
    doc['editor'].append(f5)
    doc['editor.selection'] = doc['selection']
    doc['editor.selection'].append("one")
    """
