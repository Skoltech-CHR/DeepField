"""Decorators."""
import inspect
from functools import wraps
from textwrap import dedent
import functools
import numpy as np
from anytree import PreOrderIter


class cached_property:  # pylint: disable=invalid-name
    """Cached property decorator.

    May be used to apply additional transformations to the cached variable.

    Parameters
    ----------
    arg : function
        If the decorator is used without arguments, represents the decorated method.
        Else, represents additional transformation applied to the output of the decorated method.
        In the latter case, should have the following interface:
            arg : instance, output -> modified_output
        where `instance` is the instance of the method's class.
    modify_cache : bool, optional
        If True, applies transformation not only to the output but rather to the cached variable itself.
    """
    def __init__(self, arg, modify_cache=False):
        self._update_property(arg)
        self.modify_cache = modify_cache
        self.apply_to_output = lambda instance, out: out

    def __call__(self, arg):
        self.apply_to_output = self.property
        self._update_property(arg)
        return self

    def __get__(self, instance, cls=None):
        if self.name in instance.__dict__:
            return self._get_from_cache(instance)
        return self._compute_and_store(instance)

    def __set__(self, instance, value):
        instance.__dict__[self.name] = value

    def __delete__(self, instance):
        del instance.__dict__[self.name]

    def _update_property(self, arg):
        """Update property, its name and docstring given new property candidate."""
        self.property = arg
        self.name = arg.__name__
        self.__doc__ = arg.__doc__

    def _get_from_cache(self, instance):
        """Loads data from the cache. Modifies cache if required."""
        data = instance.__dict__[self.name]
        data = self.apply_to_output(instance, data)
        if self.modify_cache:
            instance.__dict__[self.name] = data
        return data

    def _compute_and_store(self, instance):
        """Computes data using decorated method. Stores it in the cache (modified, if required)."""
        data = self.property(instance)
        if self.modify_cache:
            data = self.apply_to_output(instance, data)
            instance.__dict__[self.name] = data
        else:
            instance.__dict__[self.name] = data
            data = self.apply_to_output(instance, data)
        return data


def apply_to_each_input(method):
    """Apply the method to each input if array of inputs is given.
    If inputs are not specified, apply to each of self.attributes.
    """
    @wraps(method)
    def decorator(self, *args, attr=None, **kwargs):
        """Returned decorator."""
        is_list = True
        if isinstance(attr, str):
            attr = (attr, )
            is_list = False
        elif attr is None:
            attr = self.attributes
            if not self.attributes:
                return None

        res = []
        for att in attr:
            res.append(method(self, *args, attr=att.upper(), **kwargs))
        if isinstance(res[0], self.__class__):
            return self
        return res if is_list else res[0]
    return decorator

def apply_to_each_segment(method, include_groups=False):
    """Apply a method to each well's segment.

    Parameters
    ----------
    method : callable
        Method to be decorated. Segment should be second argument of the method.
    include_groups : bool, optional
        If False, group nodes are not evaluated. Default to False.

    Returns
    -------
    decorator : callable
        Decorated method.
    """

    @wraps(method)
    def decorator(self, *args, **kwargs):
        """Returned decorator."""
        res = []
        for segment in PreOrderIter(self.root):
            if not include_groups and segment.is_group:
                continue
            res.append(method(self, segment, *args, **kwargs))
        if isinstance(res[0], self.__class__):
            return self
        return np.array(res)

    return decorator

def extract_actions(module):
    """Extract callable attributes with first arg specified as ``input`` from a module."""
    actions_dict = {}
    arg = None
    for (k, v) in module.__dict__.items():
        if callable(v):
            try:
                arg = inspect.getfullargspec(v).args[0]
            except (TypeError, IndexError):
                continue
            if arg == 'input':
                method = {k: (v, '.'.join([module.__name__, k]), k)}
                actions_dict.update(method)
    return actions_dict

def partialmethod(func, *frozen_args, **frozen_kwargs):
    """Wrap a method with partial application of given positional
    and keyword arguments.

    Parameters
    ----------
    func : callable
        A method to wrap.
    frozen_args : misc
        Fixed positional arguments.
    frozen_kwargs : misc
        Fixed keyword arguments.

    Returns
    -------
    method : callable
        Wrapped method.
    """
    @functools.wraps(func)
    def method(self, *args, **kwargs):
        """Wrapped method."""
        return func(self, *frozen_args, *args, **frozen_kwargs, **kwargs)
    return method

TEMPLATE_DOCSTRING = """
    Compute {description} for given data.

    Parameters
    ----------
    attr : str, optional
        Attribute to get the data from.
    args : misc
        Any additional positional arguments to ``{full_name}``.
    kwargs : misc
        Any additional named arguments to ``{full_name}``.

    Returns
    -------
    output : {out_class}
        Transformed component.
"""
TEMPLATE_DOCSTRING = dedent(TEMPLATE_DOCSTRING).strip()

def add_actions(actions_dict, template_docstring):
    """Add new actions in class.

    Parameters
    ----------
    actions_dict : dict
        A dictionary, containing names of new methods as keys and a tuple of callable,
        full name and description for each method as values.
    template_docstring : str
        A string that will be formatted for each new method from
        ``actions_dict`` using ``full_name`` and ``description`` parameters
        and assigned to its ``__doc__`` attribute.

    Returns
    -------
    decorator : callable
        Class decorator.
    """
    def decorator(cls):
        """Returned decorator."""
        for method_name, (func, full_name, description) in actions_dict.items():
            docstring = template_docstring.format(full_name=full_name,
                                                  description=description,
                                                  out_class=cls.__name__)
            method = partialmethod(cls.apply, func)
            method.__doc__ = docstring
            setattr(cls, method_name, method)
        return cls
    return decorator

def instance_check(valid):
    """Run `valid` function on the class instance before method execution."""
    def decorator(method):
        """Returned decorator."""
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            """Method wrapper."""
            if not valid(self):
                raise ValueError('State of {} is not valid for applying {}.'
                                 .format(self.__class__.__name__, method.__name__))
            return method(self, *args, **kwargs)
        return wrapper
    return decorator

def attribute_check(valid):
    """Run `valid` function on the class attribute before method execution."""
    def decorator(method):
        """Returned decorator."""
        @wraps(method)
        def wrapper(self, *args, attr=None, **kwargs):
            """Method wrapper."""
            if not valid(getattr(self, attr)):
                raise ValueError('Attribute {} is not valid for applying {}.'.format(attr, method.__name__))
            return method(self, *args, attr=attr, **kwargs)
        return wrapper
    return decorator

def ndim_check(*ndims):
    """Check that attribute's `ndim` is in a list of allowed ndims."""
    return attribute_check(lambda x: x.ndim in ndims)

def state_check(valid):
    """Check if instance state is in states allowed."""
    return instance_check(lambda x: valid(x.state))
