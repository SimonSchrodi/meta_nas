import torch.nn as nn
import math
import inspect
from models import *

def reshape_model(model: nn.Module, channels: int, n_classes: int, copy_type: str = 'Starter') -> nn.Module:
    """Reshapes input and output for the model. Note that currently this does not detect models
    which are not suited for the given input data size!

    Args:
        model (nn.Module): PyTorch model
        channels (int): input channels
        n_classes (int): number of classes

    Raises:
        NotImplementedError: reshape for model is not implemented

    Returns:
        nn.Module: model with reshaped input and output layer
    """
    def _get_member_str(model: nn.Module, first: bool):
        model_state_dict = list(model.state_dict())
        idx = 0 if first else -1
        member_str = model_state_dict[idx][:model_state_dict[idx].rfind('.')]
        return member_str

    def _reshape(model: nn.Module, member_str: str, channels:int = None, n_classes:int = None, copy_type='Copy'):
        def __build_instructions(obj, q):
            """
            Breaks down a query string into a series of actionable instructions.

            Each instruction is a (_type, arg) tuple.
            arg -- The key used for the __getitem__ or __setitem__ call on
                the current object.
            _type -- Used to determine the data type for the value of
                    obj.__getitem__(arg)

            If a key/index is missing, _type is used to initialize an empty value.
            In this way _type provides the ability to
            """
            arg = []
            _type = None
            instructions = []
            for i, ch in enumerate(q):
                if ch == "[":
                    # Begin list query
                    if _type is not None:
                        arg = "".join(arg)
                        if _type == list and arg.isalpha():
                            _type = dict
                        instructions.append((_type, arg))
                        _type, arg = None, []
                    _type = list
                elif ch == ".":
                    # Begin dict query
                    if _type is not None:
                        arg = "".join(arg)
                        if _type == list and arg.isalpha():
                            _type = dict
                        instructions.append((_type, arg))
                        _type, arg = None, []

                    _type = dict
                elif ch.isalnum() or ch == "_" or ch == '-':
                    if i == 0:
                        # Query begins with alphanum, assume dict access
                        _type = type(obj)

                    # Fill out args
                    arg.append(ch)
                else:
                    TypeError("Unrecognized character: {}".format(ch))

            if _type is not None:
                # Finish up last query
                instructions.append((_type, "".join(arg)))

            return instructions

        def _setattr(obj, query, val):
            """
            This is a special setattr function that will take in a string query,
            interpret it, add the appropriate data structure to obj, and set val.
            """
            instructions = __build_instructions(obj, query)
            for i, (_, arg) in enumerate(instructions[:-1]):
                _type = instructions[i + 1][0]
                obj = _set(obj, _type, arg)

            _type, arg = instructions[-1]
            _set(obj, _type, arg, val)

        def _set(obj, _type, arg, val=None):
            """
            Helper function for calling obj.__setitem__(arg, val or _type()).
            """
            if val is not None:
                # Time to set our value
                _type = type(val)

            if isinstance(obj, dict):
                if arg not in obj:
                    # If key isn't in obj, initialize it with _type()
                    # or set it with val
                    obj[arg] = (_type() if val is None else val)
                obj = obj[arg]
            elif isinstance(obj, list):
                n = len(obj)
                arg = int(arg)
                if n > arg:
                    obj[arg] = (_type() if val is None else val)
                else:
                    # Need to amplify our list, initialize empty values with _type()
                    obj.extend([_type() for x in range(arg - n + 1)])
                obj = obj[arg]
            elif isinstance(obj, nn.Module):
                if val is not None:
                    setattr(obj, arg, val)
                else:
                    obj = getattr(obj, arg)
            return obj

        def _getattr(obj, query):
            """
            Very similar to _setattr. Instead of setting attributes they will be
            returned. As expected, an error will be raised if a __getitem__ call
            fails.
            """
            instructions = __build_instructions(obj, query)
            for i, (_, arg) in enumerate(instructions[:-1]):
                _type = instructions[i + 1][0]
                obj = _get(obj, _type, arg)

            _type, arg = instructions[-1]
            return _get(obj, _type, arg)

        def _get(obj, _type, arg):
            """
            Helper function for calling obj.__getitem__(arg).
            """
            if isinstance(obj, dict):
                obj = obj[arg]
            elif isinstance(obj, list):
                arg = int(arg)
                obj = obj[arg]
            elif isinstance(obj, nn.Module):
                try:
                    arg = int(arg)
                    obj = obj[arg]
                except ValueError:
                    obj = getattr(obj, arg)
            return obj
        
        def __create_new_layer(old_member, copy_type: str, val:int, is_input: bool):
            def _get_default_args(func):
                signature = inspect.signature(func)
                return {
                    k: v.default
                    for k, v in signature.parameters.items()
                    if v.default is not inspect.Parameter.empty
                }

            func = getattr(nn, copy_member.__class__.__name__)
            
            # get and set default parameters
            kwargs = _get_default_args(func)
            if 'Conv2d' in copy_member.__class__.__name__:
                kwargs['in_channels'] = channels if is_input else copy_member.in_channels
                kwargs['out_channels'] = copy_member.out_channels if is_input else n_classes
                kwargs['kernel_size'] = copy_member.kernel_size
            elif copy_member.__class__.__name__ == 'Linear':
                kwargs['in_features'] = channels if is_input else copy_member.in_features
                kwargs['out_features'] = copy_member.out_features if is_input else n_classes
                if not is_input and copy_type == 'StarterTailored':
                    kwargs['in_features'] = n_classes*3 # assumes concat of tailored model!
            else:
                raise NotImplementedError
            
            if copy_type == 'Default':
                pass
            elif copy_type == 'Copy':
                if 'Conv2d' in copy_member.__class__.__name__:
                    for key in ['stride', 'padding', 'padding_mode', 'dilation', 'groups']:
                        if copy_member.__class__.__name__ == 'Conv2dSame':    
                            continue
                        kwargs[key] = getattr(copy_member, key)
                kwargs['bias'] = True if copy_member.bias is not None else False # same for linear
            elif 'Starter' in copy_type:
                if 'Conv2d' in copy_member.__class__.__name__:
                    if copy_type == 'StarterTailored':
                        kwargs['in_channels'] = 1
                    kwargs['stride'] = 1
                    kwargs['padding'] = 3
                elif copy_member.__class__.__name__ == 'Linear':
                    kwargs['bias'] = True
            else:
                raise NotImplementedError
            
            return func(**kwargs)

        
        if not ((channels and not n_classes) or (not channels and n_classes)):
            raise Exception('Not allowed')
        is_input = channels is not None
        copy_member = _getattr(model, member_str)
        new_layer = __create_new_layer(copy_member, copy_type, channels if is_input else n_classes, is_input)
        _setattr(model, member_str, new_layer)

        if not is_input and copy_type == 'StarterTailored':  # special case for tailored models
            base_member_str = _get_member_str(model.base_model_tailored, False)
            copy_member = _getattr(model.base_model_tailored, base_member_str)
            new_layer = __create_new_layer(copy_member, 'Starter', n_classes, is_input) # here starter
            _setattr(model.base_model_tailored, base_member_str, new_layer)


    # first layer
    member_str = _get_member_str(model, True) 
    _reshape(model, member_str, channels=channels, copy_type=copy_type)
    
    # last layer
    member_str = _get_member_str(model, False) 
    _reshape(model, member_str, n_classes=n_classes, copy_type=copy_type)

    return model



