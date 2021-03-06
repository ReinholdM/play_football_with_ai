# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import malib.rpc.proto.data_pb2 as proto_dot_data__pb2


class DataRPCStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Pull = channel.unary_stream(
            "/rpc.DataRPC/Pull",
            request_serializer=proto_dot_data__pb2.PullRequest.SerializeToString,
            response_deserializer=proto_dot_data__pb2.PullBuffer.FromString,
        )
        self.Push = channel.stream_unary(
            "/rpc.DataRPC/Push",
            request_serializer=proto_dot_data__pb2.PushBuffer.SerializeToString,
            response_deserializer=proto_dot_data__pb2.PushReply.FromString,
        )


class DataRPCServicer(object):
    """Missing associated documentation comment in .proto file."""

    def Pull(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")

    def Push(self, request_iterator, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")


def add_DataRPCServicer_to_server(servicer, server):
    rpc_method_handlers = {
        "Pull": grpc.unary_stream_rpc_method_handler(
            servicer.Pull,
            request_deserializer=proto_dot_data__pb2.PullRequest.FromString,
            response_serializer=proto_dot_data__pb2.PullBuffer.SerializeToString,
        ),
        "Push": grpc.stream_unary_rpc_method_handler(
            servicer.Push,
            request_deserializer=proto_dot_data__pb2.PushBuffer.FromString,
            response_serializer=proto_dot_data__pb2.PushReply.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        "rpc.DataRPC", rpc_method_handlers
    )
    server.add_generic_rpc_handlers((generic_handler,))


# This class is part of an EXPERIMENTAL API.
class DataRPC(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def Pull(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_stream(
            request,
            target,
            "/rpc.DataRPC/Pull",
            proto_dot_data__pb2.PullRequest.SerializeToString,
            proto_dot_data__pb2.PullBuffer.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )

    @staticmethod
    def Push(
        request_iterator,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.stream_unary(
            request_iterator,
            target,
            "/rpc.DataRPC/Push",
            proto_dot_data__pb2.PushBuffer.SerializeToString,
            proto_dot_data__pb2.PushReply.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )
