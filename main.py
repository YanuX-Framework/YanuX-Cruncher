from model.jsonloader import JsonLoader


def main(args):
    print("YanuX Cruncher")
    try:
        path = args[1]
    except:
        print("Usage: %s <path>" % sys.argv[0])
        sys.exit(1)

    json_loader = JsonLoader(path)
    print("Loaded: " + str(len(json_loader.json_data)) + " files")

    mac_addresses = set()

    for name, log_file in json_loader.json_data.items():
        for session in log_file["sessions"]:
            for entry in session["entries"]:
                mac_addresses.add(entry["reading"]["connectionInfo"]["bssid"])
                for wifiResults in entry["reading"]["wifiResults"]:
                    mac_addresses.add(wifiResults["macAddress"])

    print len(mac_addresses)

if __name__ == '__main__':
    import sys
    main(sys.argv)
