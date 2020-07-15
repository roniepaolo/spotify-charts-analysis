db.createUser (
  {
    user: "da-mongo",
    pwd: "pucp-da-spotify-2020",
    roles: [
      {
        role: "readWrite",
        db: "spotifydb"
      }
    ]
  }
)
