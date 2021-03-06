# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: proto/control.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


DESCRIPTOR = _descriptor.FileDescriptor(
    name="proto/control.proto",
    package="rpc",
    syntax="proto3",
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
    serialized_pb=b'\n\x13proto/control.proto\x12\x03rpc"X\n\nBeatSignal\x12\x11\n\tnode_type\x18\x01 \x01(\t\x12\x0f\n\x07node_id\x18\x02 \x01(\t\x12\x13\n\x0bnode_status\x18\x03 \x01(\t\x12\x11\n\tsend_time\x18\x04 \x01(\x01"H\n\tBeatReply\x12\x13\n\x0btarget_code\x18\x01 \x01(\t\x12\x13\n\x0b\x61\x63tion_code\x18\x02 \x01(\t\x12\x11\n\tsend_time\x18\x03 \x01(\x01\x32\x39\n\nControlRPC\x12+\n\x08HeatBeat\x12\x0f.rpc.BeatSignal\x1a\x0e.rpc.BeatReplyb\x06proto3',
)


_BEATSIGNAL = _descriptor.Descriptor(
    name="BeatSignal",
    full_name="rpc.BeatSignal",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name="node_type",
            full_name="rpc.BeatSignal.node_type",
            index=0,
            number=1,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=b"".decode("utf-8"),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="node_id",
            full_name="rpc.BeatSignal.node_id",
            index=1,
            number=2,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=b"".decode("utf-8"),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="node_status",
            full_name="rpc.BeatSignal.node_status",
            index=2,
            number=3,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=b"".decode("utf-8"),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="send_time",
            full_name="rpc.BeatSignal.send_time",
            index=3,
            number=4,
            type=1,
            cpp_type=5,
            label=1,
            has_default_value=False,
            default_value=float(0),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=28,
    serialized_end=116,
)


_BEATREPLY = _descriptor.Descriptor(
    name="BeatReply",
    full_name="rpc.BeatReply",
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name="target_code",
            full_name="rpc.BeatReply.target_code",
            index=0,
            number=1,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=b"".decode("utf-8"),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="action_code",
            full_name="rpc.BeatReply.action_code",
            index=1,
            number=2,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=b"".decode("utf-8"),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
        _descriptor.FieldDescriptor(
            name="send_time",
            full_name="rpc.BeatReply.send_time",
            index=2,
            number=3,
            type=1,
            cpp_type=5,
            label=1,
            has_default_value=False,
            default_value=float(0),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key,
        ),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax="proto3",
    extension_ranges=[],
    oneofs=[],
    serialized_start=118,
    serialized_end=190,
)

DESCRIPTOR.message_types_by_name["BeatSignal"] = _BEATSIGNAL
DESCRIPTOR.message_types_by_name["BeatReply"] = _BEATREPLY
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

BeatSignal = _reflection.GeneratedProtocolMessageType(
    "BeatSignal",
    (_message.Message,),
    {
        "DESCRIPTOR": _BEATSIGNAL,
        "__module__": "proto.control_pb2"
        # @@protoc_insertion_point(class_scope:rpc.BeatSignal)
    },
)
_sym_db.RegisterMessage(BeatSignal)

BeatReply = _reflection.GeneratedProtocolMessageType(
    "BeatReply",
    (_message.Message,),
    {
        "DESCRIPTOR": _BEATREPLY,
        "__module__": "proto.control_pb2"
        # @@protoc_insertion_point(class_scope:rpc.BeatReply)
    },
)
_sym_db.RegisterMessage(BeatReply)


_CONTROLRPC = _descriptor.ServiceDescriptor(
    name="ControlRPC",
    full_name="rpc.ControlRPC",
    file=DESCRIPTOR,
    index=0,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
    serialized_start=192,
    serialized_end=249,
    methods=[
        _descriptor.MethodDescriptor(
            name="HeatBeat",
            full_name="rpc.ControlRPC.HeatBeat",
            index=0,
            containing_service=None,
            input_type=_BEATSIGNAL,
            output_type=_BEATREPLY,
            serialized_options=None,
            create_key=_descriptor._internal_create_key,
        ),
    ],
)
_sym_db.RegisterServiceDescriptor(_CONTROLRPC)

DESCRIPTOR.services_by_name["ControlRPC"] = _CONTROLRPC

# @@protoc_insertion_point(module_scope)
