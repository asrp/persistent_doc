from pyrsistent import pmap, pvector, PClass, field
from document import get_eval, Ex, Expr
from pdb import set_trace as bp

map_type = type(pmap())

class TransformDict(PClass):
    node = field()
    doc = field()
    order = field(initial=pvector)
    dict_ = field(initial=pmap)
    def __new__(cls, node, doc, dict_=pmap(), order=None):
        order = pvector(sorted(dict_.keys())) if order is None else order
        if type(dict_) != map_type:
            dict_ = pmap(dict_)
        assert(len(dict_) == len(order))
        return PClass.__new__(cls, node=node, doc=doc, order=order,
                              dict_=dict_)

    def __setitem__(self, key, value):
        if type(value) == Ex:
            value = Expr(scope=self.doc,
                         path=(self.node, ["transforms", key])).set_value(value)
            self = self.doc[self.node + ".transforms"]
        print "T __setitem__ %s %s" % (key, value)
        if key in self:
            self.clean_expr(key)
            self = self.doc[self.node + ".transforms"]
            self.change(dict_=self.dict_.set(key, value),
                               order=self.order)
        else:
            self.change(order=self.order.append(key),
                        dict_=self.dict_.set(key, value))

    def clean_expr(self, key):
        value = self.get_raw(key)
        if type(value) == Expr:
            value.remove_deps()

    def set(self, key, value):
        return PClass.set(self, dict_=self.dict_.set(key, value),
                          order=self.order.append(key)\
                          if key not in self.dict_ else self.order)

    def __contains__(self, key):
        return self.dict_.__contains__(key)

    def __getitem__(self, key):
        return get_eval(self.dict_.__getitem__(key))

    def get(self, key, default=None):
        if key in self:
            return self[key]
        else:
            return default

    def get_raw(self, key):
        return self.dict_.__getitem__(key)

    def __delitem__(self, key):
        self.clean_expr(key)
        self.change(dict_=self.dict_.remove(key),
                    order=self.order.remove(key))

    def __repr__(self):
        return "TransformDict(%s)" % dict(self.dict_).__repr__()

    def __crepr__(self):
        return "T" + self.dict_.__repr__()

    def __iter__(self):
        return self.order.__iter__()

    def __nonzero__(self):
        return bool(self.dict_)

    def clear(self):
        # Need to check if value is an expr
        for key in self:
            self.clean_expr(key)
        self.change(dict_=pmap(), order=pvector())

    def remove(self, key):
        return PClass.set(self, dict_=self.dict_.remove(key),
                          order=self.order.remove(key))

    def change(self, **kwargs):
        new_node = PClass.set(self, **kwargs)
        if new_node.doc is not None:
            new_node.doc[new_node.node + ".transforms"] = new_node
            new_node.doc.dirty.add(new_node.node)
        return new_node
