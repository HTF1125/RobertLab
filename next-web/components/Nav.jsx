"use client";

import Link from "next/link";
import Image from "next/image";
import { useState, useEffect } from "react";

const Nav = () => {
  const isUserLoggedIn = true;
  return (
    <nav className="flex-between w-full mb-16 pt-3">
      <Link href="/" className="flex gap-2 flex-center">
        <Image
          src="/assets/images/logo.svg"
          alt="Logo"
          width={30}
          height={30}
          className="object-contain"
        />
        <p className="logo_text">Robo-Advisor</p>
      </Link>

      {/* Mobile Navigation */}
      <div className="sm:flex hidden">
        {isUserLoggedIn ? (
          <div className="flex gap-3 md:gap-5">
            <Link href="/create-prompt" className="black_btn">
              Create Post
            </Link>
            <button type="button" className="outline_btn">
              Sign Out
            </button>
            <Link href="/profile" className="">
              <Image
                src="/assets/images/logo.svg"
                alt="Profile"
                width={30}
                height={30}
                className="rounded-full"
              ></Image>
            </Link>
          </div>
        ) : (
          <>
          </>
        )}
      </div>
    </nav>
  );
};

export default Nav;
