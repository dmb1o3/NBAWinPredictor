"use client";

import Image from "next/image";
import Link from "next/link";
import { useState } from "react";

function NavBar() {
  const [navbar, setNavbar] = useState(false);
  return (
    <div>
      <nav className="fixed top-0 left-0 right-0 z-10 w-full bg-grid-top">
        <div className="justify-between px-4 mx-auto md:items-center md:flex md:max-h-[90px]">
          <div>
            <div className="flex items-center justify-between px-4">
              <Link href="/" className="deskContent">
                <Image
                  src="/logo.png"
                  width={90}
                  height={0}
                  alt="logo"
                />
              </Link>
            </div>
          </div>
          <div>
            <div
              className={`flex-1 justify-self-center pb-3 mt-8 md:block md:pb-0 md:mt-0 ${
                navbar
                  ? "p-8 md:p-0 block shadow-xl md:shadow-none rounded-md"
                  : "hidden"
              }`}
            >
              <ul className="py-12 md:flex">
                <li className="pr-3 py-2 pb-2 text-base text-center text-white md:border-b-0 md:hover:text-blue-600 md:hover:bg-transparent">
                  <Link href="/player-stats" onClick={() => setNavbar(!navbar)}>
                    Player
                  </Link>
                </li>
                <li className="py-2 pb-2 pr-3 text-base text-center text-white md:border-b-0 md:hover:text-blue-600 md:hover:bg-transparent">
                  <Link href="/team-stats" onClick={() => setNavbar(!navbar)}>
                    Team
                  </Link>
                </li>
                <li className="py-2 pr-3 pb-2 text-base text-center text-white md:border-b-0 md:hover:text-blue-600 md:hover:bg-transparent">
                  <Link href="/predictions" onClick={() => setNavbar(!navbar)}>
                    Predictions
                  </Link>
                </li>
              </ul>
            </div>
          </div>
        </div>
      </nav>
    </div>
  );
}

export default NavBar;