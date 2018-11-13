import tarfile
import io


with tarfile.open("data/metal-data.tar") as archive:
    for fl in archive:
        if fl.name.endswith(".bz2"):
            try:
                with tarfile.open(fileobj=io.BytesIO(fl.tobuf())) as data_file:
                    for sub_fl in data_file:
                        print(sub_fl.name)

            except Exception as e:
                raise e
