from pathlib import Path
from typing import List


def unpack_highlight_proposals(results, threshold: float):
    normalized_results = (results - results.min()) / (results.max() - results.min())

    # filter out values lower than threshold
    result_dict = {}
    for idx in range(len(normalized_results)):
        frame = 9 * idx + 3
        if normalized_results[idx] >= threshold:
            result_dict[frame] = normalized_results[idx]
    return result_dict


def generate_json(
    date: str,
    filename: str,
    highlight_proposals: dict,
    video_path: str,
    output_path: str,
):
    relative_path = Path(video_path).name
    Path(output_path).mkdir(exist_ok=True)

    with open(Path(output_path).joinpath(f"{date}_{filename}.json"), "w") as fp:
        json.dump(highlight_proposals, fp)
    fp.close()


def generate_fcpxml(
    date: str,
    filename: str,
    file_info: List,
    highlight_proposals: dict,
    video_path: str,
    output_path: str,
):
    relative_path = Path(video_path).name
    Path(output_path).mkdir(exist_ok=True)

    has_video = 1 if any(info["codec_type"] == "video" for info in file_info) else 0
    has_audio = 1 if any(info["codec_type"] == "audio" for info in file_info) else 0

    # input file must have video
    assert has_video == 1

    if has_video:
        video_info = list(filter(lambda x: x["codec_type"] == "video", file_info))[0]
        frame_rate = float(video_info["r_frame_rate"].split("/")[0]) / float(
            video_info["r_frame_rate"].split("/")[1]
        )
        frame_rate = round(frame_rate)
        width = int(video_info["width"])
        height = int(video_info["height"])
        duration = round(float(video_info["duration"]))

    if has_audio:
        audio_info = list(filter(lambda x: x["codec_type"] == "audio", file_info))[0]

    # filter out proposals closer than min distance
    min_distance = frame_rate * 20

    keys = list(highlight_proposals.keys())
    keys.sort()

    def is_complete(target):
        return all(
            target[idx + 1] - target[0] >= min_distance
            for idx in range(len(target) - 1)
        )

    filtered_keys = []

    while not is_complete(keys):
        for j in range(len(keys) - 1, 0, -1):
            if keys[j] - keys[0] < min_distance:
                del keys[j]
        if len(keys) > 0:
            filtered_keys.append(keys.pop(0))

    filtered_keys.sort()

    # all proposals before 10 seconds yield the same output
    # prune all but one
    requires_start_proposal = any(key < 11 * frame_rate for key in filtered_keys)
    while any(key < 11 * frame_rate for key in filtered_keys):
        del filtered_keys[0]

    if requires_start_proposal:
        filtered_keys.append(0)
    filtered_keys.sort()

    with open(Path(output_path).joinpath(f"{date}_{filename}.fcpxml"), "w") as fp:
        # basic headers
        fp.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        fp.write("<!DOCTYPE fcpxml>\n")
        fp.write("\n")
        fp.write('<fcpxml version="1.10">\n')
        fp.write("\n")

        # video info
        fp.write("<!-- Resources -->\n")
        fp.write("    <resources>\n")
        fp.write(
            f'        <format id="r1" name="FFVideoFormat{height}p{frame_rate}"/>\n'
        )
        fp.write(
            f'        <asset id="r2" format="r1" hasVideo="{has_video}" hasAudio="{has_audio}">\n'
        )
        fp.write(
            f'            <media-rep kind="original-media" src="{relative_path}"/>\n'
        )
        fp.write("        </asset>\n")
        fp.write("    </resources>\n")
        fp.write("\n")

        # highlight proposals
        fp.write("<!-- Events -->\n")
        fp.write(f'    <event name="{filename}_event">\n')
        fp.write(f'        <project name="{filename}_project">\n')
        fp.write("            <!-- Project Story Elements -->\n")
        fp.write(f'            <sequence format="r1">\n')
        fp.write("                 <spine>\n")

        # for key, value in filtered_proposals.items():
        start_in_seconds = max(filtered_keys[0] // frame_rate - 10, 0)
        if start_in_seconds > 0:
            fp.write(
                f'                    <asset-clip name="video" ref="r2" offset="0s" start="0s" duration="{start_in_seconds}s">\n'
            )
            fp.write("                    </asset-clip>\n")

        for idx, key in enumerate(filtered_keys):
            start_in_seconds = max(key // frame_rate - 10, 0)
            fp.write(
                f'                    <gap offset="{start_in_seconds}s" start="{start_in_seconds}s" duration="20s">\n'
            )
            fp.write(
                f'                        <asset-clip name="{idx}" lane="1" ref="r2" offset="{start_in_seconds}s" start="{start_in_seconds}s" duration="20s">\n'
            )
            fp.write(f"                        </asset-clip>\n")
            fp.write("                    </gap>")

            if idx + 1 < len(filtered_keys):
                duration_in_seconds = (
                    filtered_keys[idx + 1] // frame_rate - key // frame_rate
                )

                if duration_in_seconds > 0:
                    fp.write(
                        f'                    <asset-clip name="video" ref="r2" offset="{start_in_seconds + 20}s" start="{start_in_seconds + 20}s" duration="{duration_in_seconds-20}s">\n'
                    )
                    fp.write("                    </asset-clip>\n")

            if idx + 1 == len(filtered_keys) and start_in_seconds + 20 < duration:
                fp.write(
                    f'<asset-clip name="video" ref="r2" offset="{start_in_seconds + 20}s" start="{start_in_seconds + 20}s" duration="{duration - start_in_seconds - 20}s">\n'
                )
                fp.write("                    </asset-clip>\n")

        fp.write("                </spine>\n")
        fp.write("            </sequence>\n")
        fp.write("        </project>\n")
        fp.write("    </event>\n")

        # basic enders
        fp.write("</fcpxml>\n")

        fp.close()
