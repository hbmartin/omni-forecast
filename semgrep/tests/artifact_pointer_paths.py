def invalid_division(root, raw_pointer):
    fingerprint = raw_pointer["fingerprint"]
    # ruleid: artifact-pointer-fields-must-not-feed-paths-directly
    return root / fingerprint


def invalid_joinpath(root, raw_pointer):
    # ruleid: artifact-pointer-fields-must-not-feed-paths-directly
    return root.joinpath(raw_pointer.get("method_id"), "state.json")


def valid_division(root, validated_pointer):
    # ok: artifact-pointer-fields-must-not-feed-paths-directly
    return root / validated_pointer["fingerprint"]


def valid_joinpath(root, fingerprint):
    # ok: artifact-pointer-fields-must-not-feed-paths-directly
    return root.joinpath(fingerprint, "state.json")
