import torch.nn as nn
import math
import functools
from models import *

def reshape_model(model: nn.Module, channels: int, n_classes: int) -> nn.Module:
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
            if copy_type == 'Copy':
                if copy_member.__class__.__name__ == 'Conv2d':
                    if is_input:
                        new_layer = nn.Conv2d(
                            channels, 
                            copy_member.out_channels, 
                            kernel_size=copy_member.kernel_size, 
                            stride=copy_member.stride, 
                            padding=copy_member.padding,
                            padding_mode=copy_member.padding_mode,
                            dilation=copy_member.dilation,
                            groups=copy_member.groups,
                            bias=True if copy_member.bias is not None else False
                        )
                    else: # is_ouput
                        new_layer = nn.Conv2d(
                            copy_member.in_channels, 
                            n_classes, 
                            kernel_size=copy_member.kernel_size, 
                            stride=copy_member.stride, 
                            padding=copy_member.padding,
                            padding_mode=copy_member.padding_mode,
                            dilation=copy_member.dilation,
                            groups=copy_member.groups,
                            bias=True if copy_member.bias is not None else False
                        )
                elif copy_member.__class__.__name__ == 'Conv2dSame':
                    if is_input:
                        new_layer = Conv2dSame(
                            channels, 
                            copy_member.out_channels, 
                            kernel_size=copy_member.kernel_size, 
                            stride=copy_member.stride, 
                            padding=copy_member.padding,
                            dilation=copy_member.dilation,
                            groups=copy_member.groups,
                            bias=True if copy_member.bias is not None else False
                        )
                    else: # is_ouput
                        new_layer = Conv2dSame(
                            copy_member.in_channels, 
                            n_classes, 
                            kernel_size=copy_member.kernel_size, 
                            stride=copy_member.stride, 
                            padding=copy_member.padding,
                            padding_mode=copy_member.padding_mode,
                            dilation=copy_member.dilation,
                            groups=copy_member.groups,
                            bias=True if copy_member.bias is not None else False
                        )
                elif copy_member.__class__.__name__ == 'Linear':
                    new_layer = nn.Linear(
                        copy_member.in_features, 
                        n_classes, 
                        bias=True if copy_member.bias is not None else False
                    )
            elif copy_type == 'Starter': # same as in starter kit
                if copy_member.__class__.__name__ == 'Conv2d':
                    if is_input:
                        new_layer = nn.Conv2d(
                            channels, 
                            copy_member.out_channels, 
                            kernel_size=copy_member.kernel_size, 
                            stride=1, 
                            padding=3
                        )
                    else: # is_ouput
                        new_layer = nn.Conv2d(
                            copy_member.in_channels,
                            n_classes, 
                            kernel_size=copy_member.kernel_size, 
                            stride=1, 
                            padding=3
                        )
                elif copy_member.__class__.__name__ == 'Linear':
                    new_layer = nn.Linear(
                        copy_member.in_features, 
                        n_classes, 
                        bias=True
                    )
            else:
                raise NotImplementedError

            return new_layer

        
        if not ((channels and not n_classes) or (not channels and n_classes)):
            raise Exception('Not allowed')
        is_input = channels is not None
        copy_member = _getattr(model, member_str)
        new_layer = __create_new_layer(copy_member, copy_type, channels if is_input else n_classes, is_input)
        _setattr(model, member_str, new_layer)


    # first layer
    member_str = _get_member_str(model, True) 
    _reshape(model, member_str, channels=channels)
    
    # last layer
    member_str = _get_member_str(model, False) 
    _reshape(model, member_str, n_classes=n_classes)

    return model